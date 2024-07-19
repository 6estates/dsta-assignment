from assignment_1.assignment_1_2.base_db import BaseDB
from assignment_1.assignment_1_2.elasticsearch_db import setup_elasticsearch_db
from assignment_1.assignment_1_2.env import ES_URL, MILVUS_HOST, MILVUS_PORT
from assignment_1.assignment_1_2.hybrid_db import setup_hybrid_db
from assignment_1.assignment_1_2.milvus_db import create_embedding, setup_milvus_db


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


def context_search(table_names,
                    context_question,
                    top_k=2,
                    llm_embedding_name='text-embedding-3-large',
                    sort_results=True,
                    selected_pages=None,
                    from_page=None,
                    to_page=None,
                    db=None,
                    db_type='hybrid',
                    milvus_database=None):

    context_question = context_question.strip()

    db_manager = None
    if db is None:

        embeddings = create_embedding(True, llm_embedding_name)

        db_manager = create_db_manager(
            db_type=db_type,
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT,
            milvus_database=milvus_database,
            es_url=ES_URL,
            collection_or_index_names=table_names,
            embedding=embeddings,
            overwrite=False
        )
        db: BaseDB = db_manager.__enter__()

    from_to_limit = dict(from_id=from_page, to_id=to_page)

    if selected_pages:
        results = db.select(selected_pages, sort_results=sort_results, **from_to_limit)
    elif (all_results_trial := db.return_all_if_possible(top_k=top_k, **from_to_limit)) is not None:
        print(all_results_trial)
        results = all_results_trial
    else:
        results = db.search(
            query=context_question, top_k=top_k, sort_results=sort_results, **from_to_limit
        )

    if db_manager:
        db_manager.__exit__(None, None, None)

    return results


def wrap_retrieval_only_result(contexts):
    return {
        'value': [
            {
                'page': c.metadata[BaseDB.page_field],
                'index': c.metadata[BaseDB.name_field],
                'score': c.metadata[BaseDB.score_field],
                'source': c.metadata[BaseDB.source_field],
                'text': c.page_content
            }
            for c in contexts
        ]
    }
