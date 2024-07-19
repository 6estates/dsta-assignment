import copy
from datetime import datetime
import re
import openai
import pydash
from langchain.schema import Document
from assignment_1.assignment_1_2.base_db import BaseDB


def compose_page_contexts(contexts):
    contexts = copy.deepcopy(contexts)

    document_name_mapping = {}
    names = {c.metadata[BaseDB.name_field] for c in contexts}
    if len(names) > 1:
        document_name_mapping = {name: f' of Document{did}' for did, name in enumerate(names, start=1)}

    # compose
    for context in contexts:
        context_page = context.metadata[BaseDB.page_field]
        context_document_suffix = document_name_mapping.get(context.metadata[BaseDB.name_field], "")
        context.page_content = f'Page {context_page}{context_document_suffix}: {context.page_content}\n'

    return contexts


def create_page_context_augmented_query(question, content_page):
    if content_page:
        request = 'Given the context pages, find the most proper answer to the question.'
        request += ' If no answer is found, return "Answer Not Found"\n\n'
        content_page = compose_page_contexts(content_page)
        content_page = sorted(
            content_page, key=lambda one: (one.metadata[BaseDB.name_field], one.metadata[BaseDB.page_field])
        )

        for context in content_page:
            request += context.page_content
    
        request += f'\nQuestion: {question}'

    else:
        request = question
        
    return request


def chat_complete(model,
                input_text,
                temperature=0,
                stream=True,
                url=None,
                key=None):
    chunks = []
    txt = ''
    messages = []
    system_message = {
        'role': 'system',
        'content': f'You are ChatGPT, a large language model trained by OpenAI, '
            f'based on the GPT-3.5 architecture. '
            f'Current date: {datetime.today().strftime("%Y-%m-%d")}'
    }
    messages.append(system_message)

    msg_content = {
        'role': 'user',
        'content': input_text
    }
    messages.append(msg_content)

    openai.api_key = key
    if url:
        openai.api_base = url
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream,
        temperature=temperature
    )

    if stream:
        for chunk in response:
            for msg in chunk['choices']:
                delta = msg['delta'] if 'delta' in msg else msg.get('message')
                if 'content' in delta:
                    chunks.append(delta['content'])
        txt = ''.join(chunks)
    else:
        # for non-stream mode, not in use
        msg = pydash.get(response, ['data', 'choices', 0, 'message'], [])
        if len(msg):
            txt = msg[0]['content']
    return txt


def llm_retrieval_qa(request, model=None, backend='openai', url=None, key=None):
    try:
        api_base = openai.api_base
        response = chat_complete(
            model,
            request,
            temperature=0.2,
            stream=True,
            url=url if backend=='triton' else None,
            key=key if backend=="openai" else "EMPTY"
        )
        
        # To reset the openai api info
        openai.api_base = api_base
        openai.api_key = key
        
    except openai.InvalidRequestError as e:
        raise Exception(f"Context length too long, try reduce top_k and try again. OpenAI original error: {e}")
    return response


def string_to_table(markdown_table_string):
    table_string = markdown_table_string

    # Split table into rows
    rows = table_string.split('\n')

    # Remove leading/trailing whitespace and vertical bars from each row
    rows = [row.strip().strip('|') for row in rows if '|' in row]

    # Remove empty rows and neck row
    rows = [row for row in rows if row and not re.search(r'^[|\-: ]+$', row)]

    # Split rows into cells and remove leading/trailing whitespace
    cell_objs = [[{"text": c.strip()} for c in re.split(r'\s*\|\s*', row)] for row in rows]

    header = [o['text'] for o in cell_objs[0]]

    filled_cell_objs = []
    for row in cell_objs:
        if len(row) >= len(header):
            filled_cell_objs.append(row[:len(header)])
        else:
            row.extend([{'text': ''}] * (len(header) - len(row)))
            filled_cell_objs.append(row)

    return {'template': header, 'cell_objs': filled_cell_objs}


def parse_item(result, result_type):
    # if result_type == 'table':
    #     return string_to_table(result)
    # else:
    #     return result
    return result


def page_context_qa(question,
                    context_documents,
                    expected_answer_type='string',
                    model=None,
                    backend='openai',
                    url=None,
                    key=None):
    request = create_page_context_augmented_query(question, context_documents)
    response = llm_retrieval_qa(request, model, backend, url, key)

    if 'answer not found' in response.lower() or 'answer cannot be found' in response.lower():
        return None

    return parse_item(response, expected_answer_type)


def wrap_retrieval_result(result, result_type='string', contexts=None):
    return {
        'value': result,
        'type': result_type,
        'contexts': [
            {
                'page': cxt.metadata[BaseDB.page_field],
                'index': cxt.metadata[BaseDB.name_field],
                'score': cxt.metadata[BaseDB.score_field],
                'source': cxt.metadata[BaseDB.source_field]
            } for cxt in (contexts or [])
        ]
    }


def convert_context(context_pages):
    if context_pages is None:
        return None
    context = context_pages['value']
    return_context  = []
    for content in context:
        value = Document(
            page_content=content['text'],
            metadata={
                'page': content['page'],
                'score': content['score'],
                'source': content['source'],
                'index_or_collection_name': content['index']
            }
        )
        return_context.append(value)

    return return_context
