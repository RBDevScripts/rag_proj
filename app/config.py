import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    '''
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
    '''
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'rag-kms-index')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default')
    VECTOR_DB_PATH = 'vector_db'