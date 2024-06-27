from abc import abstractmethod
from typing import List, Optional, Union, Dict
from langchain.schema import Document


class BaseDB:
    page_field = 'page'
    name_field = 'index_or_collection_name'
    score_field = 'score'
    source_field = 'source'

    @classmethod
    def inject_meta(cls, search_results: List[Document], name, scores, sources):
        for r, score, source in zip(search_results, scores, sources):
            r.metadata[cls.score_field] = score
            r.metadata[cls.source_field] = source
            if name is not None:
                r.metadata[cls.name_field] = name

        return search_results

    @classmethod
    def filter_selected_ids(cls, pages: List[int], from_page: Optional[int], to_page: Optional[int]) -> List[int]:
        if from_page is not None:
            pages = [idx for idx in pages if idx >= from_page]
        if to_page is not None:
            pages = [idx for idx in pages if idx <= to_page]
        return pages

    def __init__(self, setup_status):
        self.setup_status = setup_status

    def get_setup_status(self):
        return self.setup_status

    @abstractmethod
    def insert_documents(self, documents):
        raise NotImplementedError

    @abstractmethod
    def get_total_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def select_from_to(
        self, from_id: Optional[Union[int, Dict[str, int]]] = None, to_id: Optional[Union[int, Dict[str, int]]] = None
    ) -> List[Document]:
        raise NotImplementedError

    def return_all_if_possible(
        self,
        top_k: int = 2,
        from_id: Optional[Union[int, Dict[str, int]]] = None,
        to_id: Optional[Union[int, Dict[str, int]]] = None
    ) -> Optional[List[Document]]:
        """
        If top_k is larger than number of stored entities, return all
        """
        total_count = self.get_total_count()
        if from_id is None and to_id is None:
            if total_count <= top_k:
                return self.select_from_to()
        else:
            search_results = self.select_from_to(from_id, to_id)
            if len(search_results) <= top_k:
                return search_results

        return None

    @abstractmethod
    def search_only(
        self,
        query: str,
        top_k: int = 2,
        from_id: Optional[Union[int, Dict[str, int]]] = None,
        to_id: Optional[Union[int, Dict[str, int]]] = None
    ) -> List[Document]:
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 2,
        sort_results: bool = True,
        from_id: Optional[Union[int, Dict[str, int]]] = None,
        to_id: Optional[Union[int, Dict[str, int]]] = None
    ):
        search_results = self.search_only(query, top_k, from_id=from_id, to_id=to_id)
        search_results = [one_result[0] for one_result in search_results]
        if sort_results:
            search_results = sorted(search_results, key=lambda r: r.metadata[self.page_field])
        return search_results

    @abstractmethod
    def select(self, ids: List[int], sort_results: bool = True, from_id=None, to_id=None):
        raise NotImplementedError

