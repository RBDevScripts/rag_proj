import os
import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self, path):
        self.path = path
        self.embeddings = OpenAIEmbeddings()
        self._create_store()

    def _create_store(self):
        self.vector_store = Chroma(
            persist_directory=self.path,
            embedding_function=self.embeddings
        )

    def reset_store(self, hard=False):
        # Replace-mode indexing: remove old vectors before a new upload.
        if hard:
            try:
                self.vector_store.delete_collection()
            except Exception:
                pass

            if os.path.isdir(self.path):
                for name in os.listdir(self.path):
                    item_path = os.path.join(self.path, name)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        try:
                            os.remove(item_path)
                        except OSError:
                            pass
        else:
            self.vector_store.delete_collection()

        self._create_store()

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