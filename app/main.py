from flask import Flask, request, render_template, jsonify
from models.vector_store import VectorStore
from services.storage_service import S3Storage
from services.llm_service import LLMService
from config import Config
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import logging
import uuid


app = Flask(__name__)
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)
current_doc_id = None

@app.route('/')
def index():
    global current_doc_id
    # Reset active document context on browser reload/new session.
    current_doc_id = None
    llm_service.memory.clear()
    return render_template('index.html')


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_document(file):
    """Process document based on file type and return text chunks"""
    doc_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file temporarily
        file.save(temp_path)
        
        # Process based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
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


@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        global current_doc_id
        logger.debug("Upload endpoint called")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        if not file.filename.endswith(('.txt', '.pdf')):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({'error': 'Only .txt and .pdf files are supported'}), 400

        logger.debug(f"Processing file: {file.filename}")
        
        # Process the document
        try:
            text_chunks, doc_id = process_document(file)
            logger.debug(f"Document processed into {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500

        # Upload to S3
        try:
            file.seek(0)  # Reset file pointer
            storage_service.upload_file(file, file.filename)
            logger.debug("File uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return jsonify({'error': f'Error uploading to S3: {str(e)}'}), 500

        # Add to vector store
        try:
            vector_store.reset_store(hard=True)
            vector_store.add_documents(text_chunks, doc_id=doc_id)
            current_doc_id = doc_id
            llm_service.memory.clear()
            logger.debug("Documents added to vector store")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            return jsonify({'error': f'Error adding to vector store: {str(e)}'}), 500

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'chunks_processed': len(text_chunks)
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    if not current_doc_id:
        return jsonify({'error': 'Upload a document before asking questions'}), 400

    try:
        result = llm_service.get_response(data['question'], doc_id=current_doc_id)
        return jsonify({
            'response': result.get('answer', ''),
            'citations': result.get('citations', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug= True)
