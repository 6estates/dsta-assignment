import logging
import time
from contextlib import contextmanager
from typing import ContextManager, List, Optional
import elasticsearch
from elasticsearch import TransportError
from elasticsearch.client.indices import IndicesClient
from langchain.schema.document import Document
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.elasticsearch import BaseRetrievalStrategy
from assignment_1.assignment_1_2.base_db import BaseDB

logger = logging.getLogger(__name__)


def retry_on_429(max_retries):
    assert max_retries >= 0

    def decorator_func(func):
        def wrapper(*args, **kwargs):
            retry = -1
            while retry < max_retries:
                try:
                    return func(*args, **kwargs)
                except TransportError as te:
                    if te.status_code != 429 or retry == max_retries - 1:
                        raise
                    time.sleep(2 ** retry)
                    retry += 1
        return wrapper
    return decorator_func


class SkipVectorSearchStrategy(BaseRetrievalStrategy):
    def query(self, *args, **kwargs):
        new_query_body = {
            'query': {'bool': {'must': [{'match': {'text': kwargs.get('query')}}], 'filter': kwargs.get('filter', [])}}
        }
        return new_query_body

    def index(self, *args, **kwargs):
        return {'mappings': {}}

    def require_inference(self):
        return False


class CompatibleIndicesClient(IndicesClient):
    def create(self, index, **kwargs):
        super().create(index, body=kwargs)


class CompatibleElasticsearchClient:
    def __init__(self, client):
        self.client = client
        self.indices = CompatibleIndicesClient(client)

    def __getattr__(self, item):
        if item in ['search', 'indices']:
            return getattr(self, item)
        else:
            return getattr(self.client, item)

    def search(self, index, **kwargs):
        source = kwargs.pop('source')
        return self.client.search(body=kwargs, index=index, _source=source)


class CustomElasticsearchStore(ElasticsearchStore):
    @classmethod
    def if_index_exist(cls, es_url, index_name):
        tmp_client = cls.connect_to_elasticsearch(es_url=es_url)
        result = tmp_client.indices.exists(index_name)
        tmp_client.close()
        return result

    @classmethod
    def delete_index(cls, es_url, index_name):
        tmp_client = cls.connect_to_elasticsearch(es_url=es_url)
        tmp_client.indices.delete(index_name)
        tmp_client.close()

    @classmethod
    @retry_on_429(max_retries=2)
    def count_words(cls, es_url, text):
        tmp_client = cls.connect_to_elasticsearch(es_url=es_url)
        result = tmp_client.indices.analyze(body={'text': text})
        return len(result['tokens'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if elasticsearch.__version__[0] < 8:
            self.client = CompatibleElasticsearchClient(self.client)

    def get_total_count(self):
        return self.client.count(index=self.index_name)['count']

    def field_query(self, query_dict, top_k):
        def create_field_query(field_query):
            def query_fn(*args):
                return field_query
            return query_fn

        results = self.similarity_search(query='dummy', k=top_k, custom_query=create_field_query(query_dict))
        return results

    def create_index(self, overwrite):
        index_exist = self.client.indices.exists(self.index_name)

        if not overwrite and index_exist:
            return 'skipped'

        if overwrite and index_exist:
            logger.info(f"Previous index {self.index_name} has been dropped!")
            self.client.indices.delete(index=self.index_name)

        self._create_index_if_not_exists(self.index_name)

        return 'success'

    def close(self):
        self.client.close()


class ElasticsearchDB(BaseDB):
    prefix = 'rag'
    page_field = 'page'

    def __init__(self, es_url, index_name, overwrite):
        self.es_url = es_url
        self.ori_index_name = index_name
        self.index_name = get_elasticsearch_index_name(index_name)
        self.keyword_store = CustomElasticsearchStore(
            embedding=None,
            index_name=self.index_name,
            es_url=es_url,
            strategy=SkipVectorSearchStrategy()
        )

        super().__init__(self.keyword_store.create_index(overwrite))

    def insert_documents(self, documents):
        added_list = self.keyword_store.add_documents(documents, create_index_if_not_exists=False)
        logger.info(f"{len(added_list)} entities added in INDEX {self.index_name}")

    @retry_on_429(max_retries=2)
    def get_total_count(self):
        return self.keyword_store.get_total_count()

    def compose_filters(self, from_id: Optional[int] = None, to_id: Optional[int] = None) -> List[dict]:
        if from_id is None and to_id is None:
            condition = {'gte': 0}
        elif from_id is not None and to_id is not None:
            condition = {'gte': from_id, 'lte': to_id}
        elif from_id is not None:
            condition = {'gte': from_id}
        else:
            condition = {'lte': to_id}

        return [{'range': {f'metadata.{self.page_field}': condition}}]

    @retry_on_429(max_retries=2)
    def select_from_to(self, from_id: Optional[int] = None, to_id: Optional[int] = None) -> List[Document]:
        num_entities = self.get_total_count()
        if from_id is None and to_id is None:
            range_condition = {'gte': 0}
        elif from_id is not None and to_id is not None:
            range_condition = {'gte': from_id, 'lte': to_id}
        elif from_id is not None:
            range_condition = {'gte': from_id}
        else:
            range_condition = {'lte': to_id}
        page_query = {'query': {'range': {f'metadata.{self.page_field}': range_condition}}}
        all_results = self.keyword_store.field_query(page_query, num_entities)
        all_results = self.inject_meta(
            all_results, self.ori_index_name, [1] * len(all_results), ['keyword'] * len(all_results)
        )
        search_results = sorted(all_results, key=lambda r: r.metadata[self.page_field])
        return search_results

    @retry_on_429(max_retries=2)
    def return_all_if_possible(
        self, top_k: int = 2, from_id: Optional[int] = None, to_id: Optional[int] = None
    ) -> Optional[List[Document]]:
        return super().return_all_if_possible(top_k, from_id, to_id)

    @retry_on_429(max_retries=2)
    def search_only(
        self, query: str, top_k: int = 2, from_id: Optional[int] = None, to_id: Optional[int] = None
    ) -> List[Document]:
        results = self.keyword_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=None if from_id is None and to_id is None else self.compose_filters(from_id, to_id),
        )
        search_items = [one_result[0] for one_result in results]
        search_scores = [one_result[1] for one_result in results]
        search_items = self.inject_meta(search_items, self.ori_index_name, search_scores, ['keyword'] * len(results))
        results = list(zip(search_items, search_scores))
        return results

    @retry_on_429(max_retries=2)
    def select(
        self, ids: List[int], sort_results: bool = True, from_id: Optional[int] = None, to_id: Optional[int] = None
    ):
        ids = self.filter_selected_ids(ids, from_id, to_id)

        if not ids:
            return []

        page_query = {'query': {'terms': {f'metadata.{self.page_field}': ids}}}
        search_results = self.keyword_store.field_query(page_query, len(ids))
        search_results = self.inject_meta(
            search_results, self.ori_index_name, [1] * len(search_results), ['keyword'] * len(search_results)
        )
        if sort_results:
            search_results = sorted(search_results, key=lambda r: r.metadata[self.page_field])
        return search_results

    def close(self):
        self.keyword_store.close()


def get_elasticsearch_index_name(index_name):
    return f'{ElasticsearchDB.prefix}_{index_name.lower()}'


@contextmanager
def setup_elasticsearch_db(es_url, index_name, overwrite=True) -> ContextManager[ElasticsearchDB]:
    es_db = ElasticsearchDB(es_url, index_name, overwrite=overwrite)
    try:
        logger.info('💡| Elasticsearch Database Connected')
        yield es_db
    finally:
        es_db.close()
        logger.info('🔌 | Elasticsearch Database Disconnected')
