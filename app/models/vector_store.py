from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.config import Config

class VectorStore:
    def __init__(self, path):
        self.path = path  # Kept for compatibility with existing constructor usage.
        self.namespace = Config.PINECONE_NAMESPACE
        self.index_name = Config.PINECONE_INDEX_NAME
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=self.namespace
        )

    def _ensure_index(self):
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name in existing_indexes:
            return

        self.pc.create_index(
            name=self.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_REGION)
        )

    def reset_store(self, hard=False):
        # Keep existing app behavior: clear all vectors before a new upload.
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception as exc:
            # First upload can hit a missing namespace; treat it as already empty.
            if "Namespace not found" not in str(exc):
                raise

    def add_documents(self, documents, doc_id):
        for doc in documents:
            metadata = doc.metadata or {}
            metadata["doc_id"] = doc_id
            doc.metadata = metadata
        self.vector_store.add_documents(documents)
        
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever_for_doc(self, doc_id, k=4):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k, "filter": {"doc_id": doc_id}}
        )