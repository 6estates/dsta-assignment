from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager, List, Optional
from langchain.schema import Document
from base_db import BaseDB
from elasticsearch_db import ElasticsearchDB, setup_elasticsearch_db
from milvus_db import MilvusDB, setup_milvus_db


class HybridDB(BaseDB):
    @staticmethod
    def reciprocal_rank_fusion(*list_of_list_ranks_system, constant_k=60):
        """
        Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

        Args:
        * list_of_list_ranks_system: Ranked results from different IR system.
        constant_k (int): A constant used in the RRF formula (default is 60).

        Returns:
        Tuple of list of sorted documents by score and sorted documents
        """
        # Dictionary to store RRF mapping
        rrf_map = defaultdict(float)

        # Calculate RRF score for each result in each list
        for rank_list in list_of_list_ranks_system:
            for rank, item in enumerate(rank_list, 1):
                rrf_map[item] += 1 / (rank + constant_k)

        # Sort items based on their RRF scores in descending order
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

        # Return tuple of list of sorted documents by score and sorted documents
        return sorted_items, [item for item, score in sorted_items]

    def __init__(self, milvus_db: MilvusDB, es_db: ElasticsearchDB):
        self.milvus_db = milvus_db
        self.es_db = es_db
        self.ori_index_or_collection_name = milvus_db.ori_collection_name

        setup_status = 'skipped'
        if milvus_db.get_setup_status() == es_db.get_setup_status() == 'success':
            setup_status = 'success'
        super().__init__(setup_status)

    def insert_documents(self, documents):
        self.milvus_db.insert_documents(documents)
        self.es_db.insert_documents(documents)

    def get_total_count(self):
        # use milvus to get count
        return self.milvus_db.get_total_count()

    def select_from_to(self, from_id: Optional[int] = None, to_id: Optional[int] = None) -> List[Document]:
        return self.milvus_db.select_from_to(from_id, to_id)

    def return_all_if_possible(
        self, top_k: int = 2, from_id: Optional[int] = None, to_id: Optional[int] = None
    ) -> Optional[List[Document]]:
        # use milvus to return all
        return self.milvus_db.return_all_if_possible(top_k, from_id, to_id)

    def _process_db_results(self, results, small_is_better=False):
        sorted_results = sorted(results, key=lambda item: item[1], reverse=not small_is_better)
        result_by_page = {
            one_result[0].metadata[self.page_field]: one_result for one_result in sorted_results
        }
        ranked_pages = [one_result[0].metadata[self.page_field] for one_result in sorted_results]

        return ranked_pages, result_by_page

    def search_only(
        self, query: str, top_k: int = 2, from_id: Optional[int] = None, to_id: Optional[int] = None
    ) -> List[Document]:
        milvus_results = self.milvus_db.search_only(query, top_k=top_k, from_id=from_id, to_id=to_id)
        milvus_ranked_pages, milvus_result_dict = self._process_db_results(milvus_results, small_is_better=True)

        es_results = self.es_db.search_only(query, top_k=top_k, from_id=from_id, to_id=to_id)
        es_ranked_pages, es_result_dict = self._process_db_results(es_results, small_is_better=False)

        hybrid_ranked_pages_and_scores, _ = self.reciprocal_rank_fusion(milvus_ranked_pages, es_ranked_pages)
        hybrid_items = []
        hybrid_scores = []
        hybrid_sources = []
        for page, score in hybrid_ranked_pages_and_scores:
            hybrid_scores.append(score)
            if page in milvus_result_dict:
                one_item = milvus_result_dict[page][0]
                if page in es_result_dict:
                    hybrid_sources.append('hybrid')
                else:
                    hybrid_sources.append('vector')
            else:
                one_item = es_result_dict[page][0]
                hybrid_sources.append('keyword')

            hybrid_items.append(one_item)
        hybrid_items = self.inject_meta(hybrid_items, None, hybrid_scores, hybrid_sources)
        hybrid_results = list(zip(hybrid_items, hybrid_scores))
        return hybrid_results[:top_k]

    def select(
        self, ids: List[int], sort_results: bool = True, from_id: Optional[int] = None, to_id: Optional[int] = None
    ):
        # use milvus to select
        return self.milvus_db.select(ids, sort_results=sort_results, from_id=from_id, to_id=to_id)


@contextmanager
def setup_hybrid_db(milvus_host,
                    milvus_port,
                    milvus_database,
                    es_url,
                    collection_or_index_name,
                    embedding,
                    overwrite=True
                    ) -> ContextManager[HybridDB]:
    with setup_milvus_db(
        milvus_host, milvus_port, milvus_database, collection_or_index_name, embedding, overwrite
    ) as milvus_db:
        with setup_elasticsearch_db(es_url, collection_or_index_name, overwrite=overwrite) as es_db:
            yield HybridDB(milvus_db, es_db)
