import logging
from contextlib import contextmanager
from typing import ContextManager, List, Any, Optional
from langchain.embeddings import FakeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import Milvus
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema, connections
from pymilvus.orm import utility
from env import MILVUS_HOST, MILVUS_PORT, MILVUS_OPENAI_KEY, MILVUS_DATABASE
from base_db import BaseDB

logger = logging.getLogger(__name__)


class CustomMilvus(Milvus):
    """
    Supports plain query search.
    """
    def field_query(self, expr):
        # Determine result metadata fields.
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)

        res = self.col.query(expr=expr, output_fields=output_fields)

        # Organize results.
        ret = []
        for result in res:
            meta = {x: result.get(x) for x in output_fields}
            doc = Document(page_content=meta.pop(self._text_field), metadata=meta)
            ret.append(doc)
        return ret

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        r = super().add_documents(documents, **kwargs)
        collection = self.col
        collection.flush()
        logger.info(f"{collection.num_entities} entities added in TABLE {self.collection_name}")
        return r

    def get_total_count(self):
        return self.col.num_entities


class MilvusDB(BaseDB):
    langchain_text_len = 65535
    vector_field = "langchain_vector"
    text_field = "langchain_text"
    page_field = "page"

    def __init__(self, milvus_host, milvus_port, collection_name, embedding, overwrite):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.ori_collection_name = collection_name
        self.collection_name = get_milvus_collection_name(collection_name, embedding)
        self.embedding = embedding

        # must execute before Milvus initialization
        setup_status = self.create_collection(overwrite=overwrite)

        self.vector_store = CustomMilvus(
            embedding_function=embedding,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": self.milvus_port},
            vector_field=self.vector_field,
            text_field=self.text_field,
        )

        super().__init__(setup_status)

    def insert_documents(self, documents):
        for document in documents:
            document.page_content = document.page_content[:self.langchain_text_len]
        self.vector_store.add_documents(documents)

    def get_total_count(self):
        return self.vector_store.get_total_count()

    def compose_expr(self, from_id: Optional[int] = None, to_id: Optional[int] = None) -> str:
        if from_id is None and to_id is None:
            expr = f'0 <= {self.page_field}'
        elif from_id is not None and to_id is not None:
            expr = f'{from_id} <= {self.page_field} <= {to_id}'
        elif from_id is not None:
            expr = f'{from_id} <= {self.page_field}'
        else:
            expr = f'{self.page_field} <= {to_id}'
        return expr

    def select_from_to(self, from_id: Optional[int] = None, to_id: Optional[int] = None) -> List[Document]:
        expr = self.compose_expr(from_id, to_id)
        search_results = self.vector_store.field_query(expr)
        search_results = self.inject_meta(
            search_results, self.ori_collection_name, [1] * len(search_results), ['vector'] * len(search_results)
        )
        search_results = sorted(search_results, key=lambda r: r.metadata[self.page_field])
        return search_results

    def search_only(
        self, query: str, top_k: int = 2, from_id: Optional[int] = None, to_id: Optional[int] = None
    ) -> List[Document]:
        """
        search with query in page range
        """
        search_params = {'metric_type': 'L2', 'params': {'nprobe': 128}}
        expr = self.compose_expr(from_id, to_id) if from_id is not None or to_id is not None else None
        search_results = self.vector_store.similarity_search_with_score(
            query=query, param=search_params, k=top_k, expr=expr
        )
        search_items = [one_result[0] for one_result in search_results]
        search_scores = [one_result[1] for one_result in search_results]
        search_items = self.inject_meta(
            search_items,
            name=self.ori_collection_name,
            scores=[-one_result[1] for one_result in search_results],
            sources=['vector'] * len(search_results)
        )
        search_results = list(zip(search_items, search_scores))
        return search_results

    def select(
        self, ids: List[int], sort_results: bool = True, from_id: Optional[int] = None, to_id: Optional[int] = None
    ):
        """
        select based on page range
        """
        ids = self.filter_selected_ids(ids, from_id, to_id)
        if not ids:
            return []

        ids = [str(i) for i in ids]
        expr = f'{self.page_field} in [{",".join(ids)}]'
        search_results = self.vector_store.field_query(expr)
        search_results = self.inject_meta(
            search_results, self.ori_collection_name, [1] * len(search_results), ['vector'] * len(search_results)
        )
        if sort_results:
            search_results = sorted(search_results, key=lambda r: r.metadata[self.page_field])
        return search_results

    def check_collection(self):
        return utility.has_collection(self.collection_name)

    def get_embedding_dim(self):
        if isinstance(self.embedding, FakeEmbeddings):
            return self.embedding.size

        dim_map = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        return dim_map[self.embedding.model]

    def create_collection(self, overwrite=False):
        collection_exist = self.check_collection()

        if collection_exist and not overwrite:
            return "skipped"

        elif collection_exist:
            logger.info(f"Previous collection {self.collection_name} has been dropped!")
            utility.drop_collection(self.collection_name)

        pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)

        langchain_text = FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=self.langchain_text_len)

        langchain_vector = FieldSchema(
            name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.get_embedding_dim()
        )

        meta_idx = FieldSchema(name=self.page_field, dtype=DataType.INT64)

        schema = CollectionSchema(
            fields=[langchain_text, pk, meta_idx, langchain_vector],
            description="text search",
            enable_dynamic_field=True,
        )

        collection_name = self.collection_name

        collection = Collection(name=collection_name, schema=schema, shards_num=2)

        collection.create_partition("novel")

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }

        collection.create_index(field_name="langchain_vector", index_params=index_params)

        utility.index_building_progress(self.collection_name)

        return "success"


def create_embedding(milvus_openai_embedding_enabled, model_name='text-embedding-ada-002'):
    #TODO: add extra embedding method
    if milvus_openai_embedding_enabled and MILVUS_OPENAI_KEY:
        embeddings = OpenAIEmbeddings(openai_api_key=MILVUS_OPENAI_KEY, model=model_name)
    else:
        embeddings = FakeEmbeddings(size=1536)

    return embeddings


def get_milvus_collection_name(collection_name, embedding=None, embedding_name=None):
    assert embedding or embedding_name
    if embedding is None:
        embedding = create_embedding(True, embedding_name)
    prefix_map = {'text-embedding-ada-002': '', 'text-embedding-3-small': 's_', 'text-embedding-3-large': 'l_'}
    prefix = prefix_map.get(embedding.model, '') if isinstance(embedding, OpenAIEmbeddings) else ''
    return f'{prefix}{collection_name.lower()}'


@contextmanager
def setup_milvus_db(
    milvus_host, milvus_port, database, table_name, embedding, overwrite=True
) -> ContextManager[MilvusDB]:
    connections.connect(host=milvus_host, port=milvus_port, db_name=database)

    milvus_db = MilvusDB(milvus_host, milvus_port, table_name, embedding, overwrite)
    try:
        logger.info('💡| Milvus Database Connected')
        yield milvus_db
    finally:
        connections.disconnect(database)
        logger.info('🔌 | Milvus Database Disconnected')


if __name__ == "__main__":
    # try simple index building and retrieving with milvus
    from assignment_1.assignment_1_1.data_chunk import create_document_string
    import json
    import pydash
    from pathlib import Path
    file_data = json.loads(Path("path-to—idp-doc-file").read_text())
    table_name = "jsontable11"
    milvus_openai_embedding_enabled = False

    embeddings = create_embedding(milvus_openai_embedding_enabled)
    with setup_milvus_db(
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        database=MILVUS_DATABASE,
        table_name=table_name,
        embedding=embeddings,
        overwrite=True,
    ) as milvus_db:
        milvus_doc = []
        pages = pydash.get(file_data, ["files", 0, "pages"])
        file_by_page = pydash.key_by(pages, "page")
        for page_id, page in file_by_page.items():
            if not len(page["paragraphs"]):
                continue
            raw_doc = create_document_string(page, prefix="", connector=" ")
            milvus_doc.append(Document(page_content=raw_doc[: milvus_db.langchain_text_len]))
        milvus_db.insert_documents(milvus_doc)
        results = milvus_db.search(
            query="SAMPLE QUERY", top_k=3
        )
        print(results[0])
