from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from config import Config

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
            return_messages=True
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
                memory=self.memory
            )

            response = chain({"question": query})
            return response['answer']
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I encountered an error processing your request."