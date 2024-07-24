import pydash
import json
from assignment_3.assignment_3_2.qa_search import convert_context, llm_retrieval_qa, page_context_qa, parse_item, wrap_retrieval_result
from assignment_3.assignment_3_2.retrieve import context_search, wrap_retrieval_only_result
from assignment_3.assignment_3_2.search_news import search_news


class RAGPipeline:
    def __init__(self,
                 triton_url,
                 openai_api_key,
                 google_key,
                 google_cx,
                 model=None,
                 table_names=None,
                 llm_embedding_name=None,
                 milvus_collection_name='jsonfiledb',
                 database_type='hybrid'):
        self.triton_url = triton_url
        self.openai_api_key = openai_api_key
        self.google_key = google_key
        self.google_cx = google_cx
        self.milvus_collection_name = milvus_collection_name
        self.set_model(model)
        self.set_document(table_names, llm_embedding_name)
        self.set_database_type(database_type)


    def set_model(self, model):
        if model is not None:
            if model.startswith('gpt'):
                backend = 'openai'
            else:
                backend = 'triton'
            
            self.model_info = {
                "model": model,
                "backend": backend
            }
        else:
            self.model_info = None


    def set_document(self, table_names, llm_embedding_name):
        if table_names is not None and llm_embedding_name is not None:
            table_names = [table_names]
            self.document_info = {
                "table_names": table_names,
                "llm_embedding_name": llm_embedding_name
            }
        else:
            self.document_info = None

    
    def set_database_type(self, database_type):
        if database_type is not None:
            self.database_type = database_type
        else:
            self.database_type = None


    def _retrieve(self, question, table_names, selected_pages=[], from_page=None, to_page=None,
                 database_type='hybrid', top_k=2, llm_embedding_name='text-embedding-3-large'):
        context_question = question.strip()
        context_pages = context_search(
            table_names,
            context_question,
            top_k=top_k,
            llm_embedding_name=llm_embedding_name,
            sort_results=False,
            selected_pages=selected_pages,
            from_page=from_page,
            to_page=to_page,
            db_type=database_type,
            milvus_database=self.milvus_collection_name,
        )
        return context_pages
    

    def retrieve(self, question, selected_pages=[], from_page=None, to_page=None, top_k=2):
        assert self.document_info is not None, f"Please set the document info with the set_document(table_names, llm_embedding_name) method!"

        context_pages = self._retrieve(question, self.document_info['table_names'], selected_pages, from_page,
                to_page, self.database_type, top_k, self.document_info['llm_embedding_name'])
        return wrap_retrieval_only_result(context_pages)
    

    def _qa_search(self, question, answer_type='string', model='gpt-3.5-turbo', backend='openai'):
        user_question = question.strip()
        response = llm_retrieval_qa(user_question, model, backend, self.triton_url, self.openai_api_key)
        if 'answer not found' in response.lower() or 'answer cannot be found' in response.lower():
            return None
        return parse_item(response, answer_type)
    

    def _qa_search_with_context(self, question, context_pages=None, answer_type='string', model='gpt-3.5-turbo', backend='openai'):
        user_question = question.strip()
        result = page_context_qa(
            user_question,
            context_pages,
            expected_answer_type=answer_type,
            model=model,
            backend=backend,
            url=self.triton_url,
            key=self.openai_api_key
        )
        return result
    

    def qa_search(self, question, context_pages=None):
        assert self.model_info is not None, f"Please set the model info with the set_model(model) method!"

        answer_type = 'string'
        model = self.model_info['model']
        backend = self.model_info['backend']
        if context_pages:
            context_pages = convert_context(context_pages)
            result = self._qa_search_with_context(question, context_pages, answer_type, model, backend)
            # return wrap_retrieval_result(result, result_type=answer_type, contexts=context_pages)
        else:
            result = self._qa_search(question, answer_type, model, backend)
        return result
        

    def retrieve_and_search(self, context_question, user_question,
                            selected_pages=[], from_page=None, to_page=None,
                            top_k=2):
        assert self.document_info is not None, f"Please set the document info with the set_document(table_names, llm_embedding_name) method!"
        assert self.model_info is not None, f"Please set the model info with the set_model(model) method!"
        
        answer_type='string'
        table_names = self.document_info['table_names']
        database_type = self.database_type
        llm_embedding_name = self.document_info['llm_embedding_name']
        model = self.model_info['model']
        backend = self.model_info['backend']
        
        context_pages = self._retrieve(context_question, table_names, selected_pages, from_page,
                to_page, database_type, top_k, llm_embedding_name)
        result = self._qa_search_with_context(user_question, context_pages, answer_type, model, backend)
        result = wrap_retrieval_result(result, result_type=answer_type, contexts=context_pages)
        return result['value']
    

    def news_search(self, key_words, type_num=1, sort=None, top_k=10):
        search_results = search_news(key_words, self.google_key, self.google_cx, type_num, sort, top_k)
        return search_results
    

    def filter_news(self, question):
        model = self.model_info['model']
        backend = self.model_info['backend']
        answer_type='string'
        result = self._qa_search(question, answer_type, model, backend)
        return result
    

    def show_news_result(self, news_list):
        news_list = json.loads(news_list)
        for i, news in enumerate(news_list):
            print(f"\nSearch Result {i + 1}:")
            title = pydash.get(news, ['title'], '')
            link = pydash.get(news, ['link'], '')
            snippet = pydash.get(news, ['snippet'], '')
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Snippet: {snippet}")
    

if __name__ == '__main__':
    _url = 'http://10.12.0.12:8000/v1'
    _key = 'YOUR OPENAI_API_KEY'
    _google_key = 'YOUR GOOGLE_KEY'
    _google_cx = 'YOUR GOOGLE_CX'
    _model = 'gpt-3.5-turbo'
    _table_name = 'default_table_3_2'
    _llm_embedding_name = 'text-embedding-3-large'
    _milvus_database = 'jsonfiledb'
    _database_type = 'hybrid'


    rag = RAGPipeline(_url, _key, _google_key, _google_cx, _model, _table_name,
                      _llm_embedding_name,  _milvus_database, _database_type)

    
    top_k = 2
    context_question = "What is the financial year?"
    selected_pages = []
    from_page = None
    to_page = None

    retrieve_result = rag.retrieve(context_question, selected_pages, from_page, to_page, top_k)
    print(retrieve_result)
    print("\n\n")


    context_pages ={'value': 
        [
            {
                'page': 9,
                'index': 'jsontable11',
                'score': 0.01639344262295082,
                'source': 'vector',
                'text': """
                        ANNUAL REPORT   2023   63
                        DIRECTORS’ REPORT  (cont’d)
                        AUDITORS
                        The auditors,   Messrs.  UHY, have   expressed  their willingness to continue in office.
                        Signed on   behalf  of the   Board  of Directors in accordance with a resolution of the directors: 
                        ……………………………………..
                        CHOONG LEE   AUN
                        Director
                        ……………………………………..
                        MAK SIEW   WEI
                        Director
                        Date: 25   July  2023
                        """
            },
            {
                'page': 3,
                'index': 'jsontable11',
                'score': 0.016129032258064516,
                'source': 'vector',
                'text': """
                ANNUAL REPORT   2023   57
                DIRECTORS’ 
                REPORT
                The directors   hereby  submit their report and the audited financial statements of the Group and of the Company for 
                the financial   year  ended 31 March 2023.
                PRINCIPAL ACTIVITIES
                The principal   activities  of the Company are those of investment holding and provision of management services to 
                its subsidiaries.   The  principal activities of the subsidiaries are set out in Note 11. 
                
                RESULTS
                      Group    Company
                            RM        RM
                Loss for   the  financial year                                  (82,745,163)    (2,929,551)
                Attributable to:-
                Owners of   the  Company                                   (82,745,163)    (2,929,551)
                                                                        (82,745,163)    (2,929,551)
                DIVIDEND
                No dividend   has  been paid, declared or proposed by the Company since the end of the previous financial year. The 
                directors do   not  recommend the payment of any dividend in respect of the current financial year.
                RESERVES AND   PROVISIONS
                There were   no  material transfers to or from reserves or provisions during the financial year other than those 
                disclosed in   the  financial statements.
                BAD AND   DOUBTFUL  DEBTS
                Before the   statements  of profit or loss and other comprehensive income and statements of financial position of the 
                Group and   of  the Company were made out, the directors took reasonable steps to ascertain that action had been 
                taken in   relation  to the writing off of bad debts and the making of allowance for doubtful debts and have satisfied 
                themselves that   adequate  allowance had been made for doubtful debts and there were no bad debts to be written 
                off.
                At the   date  of this report, the directors are not aware of any circumstances which would render it necessary to write 
                off any   bad  debts or the amount of allowance for doubtful debts in the financial statements of the Group and of the 
                Company inadequate   to  any substantial extent.
                """
            }
        ]
    }

    user_question = """If there is context, tell me the financial year based on the context.
    If not, tell me what year it is directly."""

    qa_result = rag.qa_search(user_question, context_pages)
    print(qa_result)
    print("\n\n")


    result = rag.retrieve_and_search(context_question, user_question, selected_pages, from_page, to_page, top_k)
    print(result)
    print("\n\n")


    context_pages = None
    qa_result = rag.qa_search(user_question, context_pages)
    print(qa_result)
    print("\n\n")


    _model = 'llama70b'
    _database_type = 'vector'
    rag.set_model(_model)
    rag.set_database_type(_database_type)

    result = rag.retrieve_and_search(context_question, user_question, selected_pages, from_page, to_page, top_k)
    print(result)
    print("\n\n")


    company_name = 'ByteDance'
    key_words = f"What is {company_name}'s upcoming activities and events?"
    type_num = 1
    sort = 'date'
    top_k = 5

    context = rag.news_search(key_words, type_num, sort, top_k)
    rag.show_news_result(context)

    question = f"""
        You are an experienced financial analyst that is capable of providing in-depth analysis of financial statements.
        Given the company name as {company_name}, and the recent news of this company in json format as:\n
        {context}
        , filter and select the most relevant news which are possibly useful for financial analyst's analysis of this company.\n
        Select maximum 2 news, and output the title, link and snippet of each news.
    """
    
    result = rag.filter_news(question)
    print(result)
