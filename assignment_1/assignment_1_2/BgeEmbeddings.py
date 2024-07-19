from typing import List
from langchain.schema.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalBgeEmbeddings(Embeddings):
    size = 384

    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path).cpu()

    def _get_embedding(self, text) -> List[float]:
        return list(self.model.encode(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

# question: would it be better to split page information when using BgeEmbedding?
