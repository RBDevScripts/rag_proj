from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from app.config import Config

class LLMService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",  # 🔥 best Groq model
            temperature=0.7
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def get_response(self, query, doc_id=None):
        try:
            if doc_id:
                retriever = self.vector_store.get_retriever_for_doc(doc_id)
            else:
                retriever = self.vector_store.vector_store.as_retriever()

            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True
            )

            response = chain.invoke({"question": query})
            citations = []
            seen = set()

            for source_doc in response.get("source_documents", []):
                metadata = source_doc.metadata or {}
                source_name = metadata.get("filename") or metadata.get("source") or "Unknown source"
                page = metadata.get("page", "N/A")
                page_content = (source_doc.page_content or "").strip().replace("\n", " ")
                snippet = page_content[:200] + ("..." if len(page_content) > 200 else "")
                key = (source_name, page)

                if key in seen:
                    continue

                seen.add(key)
                citations.append(
                    {
                        "source": source_name,
                        "page": page,
                        "snippet": snippet,
                    }
                )

            return {
                "answer": response.get("answer", ""),
                "citations": citations,
            }
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return {
                "answer": "I encountered an error processing your request.",
                "citations": [],
            }