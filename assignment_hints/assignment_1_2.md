## Suggested Answers
Note: `...` means that there is no change to the original code.

### Question 1
```Python
def create_document_string(page, prefix='### Input:\n\n', connector='\t', simple_join=True):
    ...
    
    # line 127 in assignment_1_1/data_chunk.py
    else:
        sorted_paragraphs = sorted(page['paragraphs'], key=lambda x: (x['bbox'][1], x['bbox'][0]))
        document = ' '.join([paragraph['text'] for paragraph in sorted_paragraphs])
        document = re.sub(r'\n', ' ', document)
    return document      
```

### Question 2
```Python
def build_document_index(pdf_bin, table_name, converter_engine, milvus_openai_embedding_enabled=True,
                         embedding_name='BgeEmbeddings', overwrite=False, milvus_only=False):

    embeddings = create_embedding(milvus_openai_embedding_enabled, embedding_name)
    
    if converter_engine == 'PYMUPDF':
        one_idp_doc = convert_e_pdf(pdf_bin)
    else:
        one_idp_doc = convert_scanned_pdf(pdf_bin)

    file_pages = pydash.get(one_idp_doc, ['pages'])
    if not file_pages:
        raise BadRequestProblem(detail='No content found in input file')

    doc = []
    file_by_page = pydash.key_by(file_pages, 'page')
    for page_id, page in file_by_page.items():
        # Arrange raw text by page
        if not len(page['paragraphs']):
            continue
        raw_doc = create_document_string(page, prefix='', connector=' ')
        doc.append(Document(page_content=raw_doc, metadata={'page': page_id}))
    if len(doc) == 0:
        return 'No content found in input file'

    if milvus_only:
        try:
            with setup_milvus_db(milvus_host=MILVUS_HOST,
                                 milvus_port=MILVUS_PORT,
                                 database=MILVUS_DATABASE,
                                 table_name=table_name,
                                 embedding=embeddings,
                                 overwrite=True) as milvus_db:
                status = milvus_db.get_setup_status()
                if status == 'skipped':
                    raise BadRequestProblem(
                        detail=f'The table name {table_name} has been used before.'
                               f' If you want to create a new table with the same name, request with overwrite=True.'
                    )
                milvus_db.insert_documents(doc)
                return 'Document finished index building in Milvus database'
        except Exception as e:
            return f"ERRORS when building index in Milvus: {e}"

    else:
        try:
            with setup_hybrid_db(milvus_host=MILVUS_HOST,
                                 milvus_port=MILVUS_PORT,
                                 milvus_database=MILVUS_DATABASE,
                                 es_url=ES_URL,
                                 # must be lowercase in elasticsearch
                                 collection_or_index_name=table_name.lower(),
                                 embedding=embeddings,
                                 overwrite=overwrite) as hybrid_db:

                status = hybrid_db.get_setup_status()

                if status == 'skipped':
                    raise BadRequestProblem(
                        detail=f'The table name {table_name} has been used before.'
                               f' If you want to create a new table with the same name, request with overwrite=True.'
                    )
                hybrid_db.insert_documents(doc)
                return 'Document finished index building in hybrid database'
        except Exception as e:
            return f"ERRORS when building index in Milvus/Elasticsearch: {e}"
``` 

### Question 3 
OpenAI embeddings (`text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`) should give you the most accurate result compared to `BgeEmbeddings` or `FakeEmbeddings`.

### Question 4
OpenAI embeddings (`text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`) should give you the most accurate result compared to `BgeEmbeddings` or `FakeEmbeddings`.   

You will also notice that hybrid database will give you more accurate search result as compared to Milvus database only 