import os
import tempfile
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.config import Config
from app.models.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.storage_service import S3Storage


app = FastAPI(title="Knowledge Management System")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)
current_doc_id = None

class QueryRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    global current_doc_id
    # Reset active document context on browser reload/new session.
    current_doc_id = None
    llm_service.memory.clear()
    return templates.TemplateResponse(request, "index.html", {})


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def process_document(file: UploadFile):
    """Process document based on file type and return text chunks"""
    doc_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "uploaded_file")
    
    try:
        # Save file temporarily
        with open(temp_path, "wb") as temp_file:
            temp_file.write(await file.read())
        
        # Process based on file type
        filename = (file.filename or "").lower()
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)

        # Normalize source metadata so citations show the original filename.
        for chunk in text_chunks:
            metadata = chunk.metadata or {}
            metadata["filename"] = file.filename
            chunk.metadata = metadata
        
        return text_chunks, doc_id
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        global current_doc_id
        logger.debug("Upload endpoint called")

        # Check file extension
        if not file.filename:
            logger.warning("Empty filename")
            raise HTTPException(status_code=400, detail="No file selected")

        if not file.filename.endswith(('.txt', '.pdf')):
            logger.warning(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

        logger.debug(f"Processing file: {file.filename}")
        
        # Process the document
        try:
            text_chunks, doc_id = await process_document(file)
            logger.debug(f"Document processed into {len(text_chunks)} chunks")
            if not text_chunks:
                raise ValueError("No extractable text found in document")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}") from e

        # Upload to S3
        try:
            await file.seek(0)  # Reset file pointer
            storage_service.upload_file(file.file, file.filename)
            logger.debug("File uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}") from e

        # Add to vector store
        try:
            vector_store.reset_store(hard=True)
            vector_store.add_documents(text_chunks, doc_id=doc_id)
            current_doc_id = doc_id
            llm_service.memory.clear()
            logger.debug("Documents added to vector store")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error adding to vector store: {str(e)}") from e

        return {
            'message': 'File uploaded and processed successfully',
            'chunks_processed': len(text_chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}") from e
    


@app.post("/query")
async def query(payload: QueryRequest):
    if not payload.question:
        raise HTTPException(status_code=400, detail="No question provided")
    if not current_doc_id:
        raise HTTPException(status_code=400, detail="Upload a document before asking questions")

    try:
        result = llm_service.get_response(payload.question, doc_id=current_doc_id)
        return {
            'response': result.get('answer', ''),
            'citations': result.get('citations', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
