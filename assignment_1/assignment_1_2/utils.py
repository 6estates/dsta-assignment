from pymilvus import connections
from pymilvus.orm import utility
from env import MILVUS_HOST, MILVUS_PORT, ES_URL, MILVUS_DATABASE
from elasticsearch_db import setup_elasticsearch_db, CustomElasticsearchStore, get_elasticsearch_index_name
from hybrid_db import setup_hybrid_db
from milvus_db import setup_milvus_db, get_milvus_collection_name


def create_db_manager(db_type,
                      milvus_host,
                      milvus_port,
                      milvus_database,
                      es_url,
                      collection_or_index_names,
                      embedding,
                      overwrite=False):
    if db_type == 'keyword':
        db_manager = setup_elasticsearch_db(es_url=es_url, index_name=collection_or_index_names[0], overwrite=overwrite)
    elif db_type == 'vector':
        db_manager = setup_milvus_db(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            database=milvus_database,
            table_name=collection_or_index_names[0],
            embedding=embedding,
            overwrite=overwrite
        )
    else:
        db_manager = setup_hybrid_db(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_database=milvus_database,
            es_url=es_url,
            collection_or_index_name=collection_or_index_names[0],
            embedding=embedding,
            overwrite=overwrite
        )

    return db_manager


def if_milvus_collection_exist(milvus_host, milvus_port, database, collection_names, embedding_name):
    connections.connect(host=milvus_host, port=milvus_port, db_name=database)
    for collection_name in collection_names:
        valid_collection_name = get_milvus_collection_name(collection_name, embedding_name=embedding_name)
        result = utility.has_collection(valid_collection_name)
        if not result:
            return False, collection_name
    connections.disconnect(database)
    return True, None


def if_elasticsearch_db_exist(es_url, index_names):
    for index_name in index_names:
        exist = CustomElasticsearchStore.if_index_exist(
            es_url, get_elasticsearch_index_name(index_name)
        )
        if not exist:
            return False, index_name
    return True, None


def if_collection_or_index_available(db_type, collection_or_index_names, embedding_name):

    available_in_vector_db = True
    available_in_keyword_db = True
    missing_collection = missing_index = None
    if db_type in ['vector', 'hybrid']:
        available_in_vector_db, missing_collection = if_milvus_collection_exist(
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT,
            database=MILVUS_DATABASE,
            collection_names=collection_or_index_names,
            embedding_name=embedding_name
        )
    if db_type in ['keyword', 'hybrid']:
        available_in_keyword_db, missing_index = if_elasticsearch_db_exist(
            ES_URL, collection_or_index_names
        )

    return available_in_keyword_db and available_in_vector_db, missing_collection or missing_index


def if_context_query_valid(context_query: str) -> bool:
    """
    If context_query > indices.query.bool.max_clause_count, "too_many_clauses" error can happen. To avoid it,
    we check the query length
    """
    context_len = CustomElasticsearchStore.count_words(ES_URL, context_query)
    if context_len <= 1024:
        return True

    return False


def drop_milvus_collection(milvus_host, milvus_port, database, collection_name, embedding_name):
    connections.connect(host=milvus_host, port=milvus_port, db_name=database)
    try:
        valid_collection_name = get_milvus_collection_name(collection_name, embedding_name=embedding_name)
        utility.drop_collection(valid_collection_name)
    finally:
        connections.disconnect(database)


def drop_elasticsearch_index(es_url, index_name):
    CustomElasticsearchStore.delete_index(es_url, get_elasticsearch_index_name(index_name))


def drop_collection_or_index(collection_or_index_name, embedding_name):
    drop_milvus_collection(
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        database=MILVUS_DATABASE,
        collection_name=collection_or_index_name,
        embedding_name=embedding_name
    )
    drop_elasticsearch_index(ES_URL, collection_or_index_name)

    return True

