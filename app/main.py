import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, HttpUrl

from app.config import Config
from app.models.vector_store import VectorStore
from app.services.llm_service import LLMService


app = FastAPI(title="Knowledge Management System")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
vector_store = VectorStore(Config.VECTOR_DB_PATH)
llm_service = LLMService(vector_store)
current_doc_id = None

EXCLUDED_DIRECTORIES = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".idea",
    ".vscode",
}
ALLOWED_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".ps1",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env.example",
    ".md",
    ".txt",
}
ALLOWED_FILENAMES = {
    "dockerfile",
    "makefile",
    "readme",
    "readme.md",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pyproject.toml",
}
MAX_FILE_SIZE_BYTES = 1_000_000
MAX_FILES_TO_INDEX = 200

class QueryRequest(BaseModel):
    question: str


class RepoRequest(BaseModel):
    repo_url: HttpUrl


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


def _normalize_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) repository URLs are supported")
    if not parsed.netloc:
        raise ValueError("Repository URL is invalid")
    if not parsed.path.endswith(".git"):
        return repo_url.rstrip("/") + ".git"
    return repo_url


def _build_repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path = parsed.path.rstrip("/")
    repo_name = Path(path).stem or "repository"
    return repo_name


def _should_include_file(file_path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORIES for part in file_path.parts):
        return False
    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return False

    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    if suffix in ALLOWED_FILE_EXTENSIONS:
        return True
    if name in ALLOWED_FILENAMES:
        return True
    return False


def _collect_repo_documents(repo_dir: Path, repo_url: str):
    documents = []
    indexed_files = 0

    for file_path in repo_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if not _should_include_file(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug("Skipping non-UTF8 file: %s", file_path)
            continue
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            continue

        if not content.strip():
            continue

        relative_path = file_path.relative_to(repo_dir).as_posix()
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "filename": relative_path,
                    "source": relative_path,
                    "repo_url": repo_url,
                    "repo_name": repo_dir.name,
                    "file_type": file_path.suffix.lower() or file_path.name,
                },
            )
        )
        indexed_files += 1
        if indexed_files >= MAX_FILES_TO_INDEX:
            break

    return documents


def process_repository(repo_url: str):
    """Clone a repository, read supported files, and return text chunks."""
    doc_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())
    repo_slug = _build_repo_slug(repo_url)
    clone_dir = temp_dir / repo_slug
    normalized_url = _normalize_repo_url(repo_url)

    try:
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", normalized_url, str(clone_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_result.returncode != 0:
            raise ValueError(clone_result.stderr.strip() or "Failed to clone repository")

        documents = _collect_repo_documents(clone_dir, repo_url)
        if not documents:
            raise ValueError("No supported text/code files found in repository")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
        )
        text_chunks = text_splitter.split_documents(documents)

        for chunk in text_chunks:
            metadata = chunk.metadata or {}
            metadata["repo_url"] = repo_url
            metadata["repo_name"] = repo_slug
            chunk.metadata = metadata

        return text_chunks, doc_id, len(documents)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ingest-repo")
async def ingest_repository(payload: RepoRequest):
    try:
        global current_doc_id
        logger.debug("Repository ingestion called for %s", payload.repo_url)

        try:
            text_chunks, doc_id, indexed_files = process_repository(str(payload.repo_url))
            logger.debug("Repository processed into %s chunks", len(text_chunks))
            if not text_chunks:
                raise ValueError("No extractable text found in repository")
        except ValueError as e:
            logger.error("Invalid repository input: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("Error processing repository: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error processing repository: {str(e)}") from e

        try:
            vector_store.reset_store(hard=True)
            vector_store.add_documents(text_chunks, doc_id=doc_id)
            current_doc_id = doc_id
            llm_service.memory.clear()
            logger.debug("Repository documents added to vector store")
        except Exception as e:
            logger.error("Error adding to vector store: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error adding to vector store: {str(e)}") from e

        return {
            'message': 'Repository indexed successfully',
            'chunks_processed': len(text_chunks),
            'files_indexed': indexed_files,
            'repo_url': str(payload.repo_url),
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
        raise HTTPException(status_code=400, detail="Index a repository before asking questions")

    try:
        result = llm_service.get_response(payload.question, doc_id=current_doc_id)
        return {
            'response': result.get('answer', ''),
            'citations': result.get('citations', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
