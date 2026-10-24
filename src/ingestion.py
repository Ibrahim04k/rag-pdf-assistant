"""
Module: src/ingestion.py
Description: Document ingestion, text extraction with page metadata preservation,
             recursive semantic chunking, and local FAISS vector indexing.
"""

import os
import io
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field

# PDF Parsing
try:
    import pypdf
    from pypdf import PdfReader
except ImportError:
    pypdf = None
    PdfReader = None

# LangChain Document Schema
try:
    from langchain_core.documents import Document
except ImportError:
    # Lightweight fallback Document definition
    @dataclass
    class Document:  # type: ignore
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

# Text Splitter with robust fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

# Vector Store and Embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain_faiss import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


class FallbackRecursiveSplitter:
    """
    Pure Python recursive text splitter fallback if LangChain splitter is unavailable.
    Recursively splits on paragraphs, newlines, sentences, and words while preserving overlap.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                # Try to find natural breaking point from separators
                split_pos = -1
                for sep in self.separators:
                    pos = text.rfind(sep, start + (self.chunk_size // 2), end)
                    if pos != -1:
                        split_pos = pos + len(sep)
                        break
                if split_pos != -1:
                    end = split_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            # Step forward by chunk_size - chunk_overlap
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result_docs = []
        for doc in documents:
            text_splits = self.split_text(doc.page_content)
            for i, split in enumerate(text_splits):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                result_docs.append(Document(page_content=split, metadata=meta))
        return result_docs


def clean_extracted_text(raw_text: str) -> str:
    """
    Cleans raw PDF text by removing excessive whitespace, resolving hyphenated line breaks,
    and stripping unprintable control characters.
    """
    if not raw_text:
        return ""
    # Merge hyphenated line breaks: "informa-\ntion" -> "information"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
    # Replace multiple vertical whitespace characters with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace tabs and multiple horizontal spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_text_from_pdf(
    file_source: Union[str, io.BytesIO, Any],
    source_name: Optional[str] = None
) -> List[Document]:
    """
    Extracts text page-by-page from a PDF file path or file stream (e.g. Streamlit UploadedFile),
    preserving metadata including the source filename and 1-indexed page number.

    Args:
        file_source: Filepath string or byte stream.
        source_name: Human-readable filename for citations.

    Returns:
        List of LangChain Document objects with page content and metadata.
    """
    if PdfReader is None:
        raise ImportError("pypdf is required for document ingestion. Run: pip install pypdf")

    # Determine filename
    if source_name:
        filename = source_name
    elif isinstance(file_source, str):
        filename = os.path.basename(file_source)
    elif hasattr(file_source, "name"):
        filename = getattr(file_source, "name", "document.pdf")
    else:
        filename = "document.pdf"

    documents: List[Document] = []

    # Read PDF
    try:
        reader = PdfReader(file_source)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = clean_extracted_text(raw_text)

            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "page": page_number,
                        "total_pages": total_pages,
                        "char_count": len(cleaned_text),
                    }
                )
                documents.append(doc)

    except Exception as e:
        raise RuntimeError(f"Error parsing PDF '{filename}': {str(e)}") from e

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits documents into overlapping semantic chunks using RecursiveCharacterTextSplitter.
    Injects unique chunk identifiers and metadata for exact citation tracing.

    Args:
        documents: List of page-level Documents.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlapping character count between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        return []

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            keep_separator=True
        )
    else:
        splitter = FallbackRecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    chunked_docs = splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifiers and metrics
    for idx, chunk in enumerate(chunked_docs):
        src = chunk.metadata.get("source", "doc")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"{src}_p{page}_c{idx+1}"
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

    return chunked_docs


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu"
) -> Any:
    """
    Initializes and caches the Hugging Face sentence-transformers embedding model.
    Configures normalization so FAISS L2 distance maps directly to Cosine Similarity.

    Args:
        model_name: Hugging Face model identifier (default: sentence-transformers/all-MiniLM-L6-v2).
        device: Computing device ('cpu' or 'cuda').

    Returns:
        HuggingFaceEmbeddings instance.
    """
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            "langchain-huggingface or langchain-community is required for vector embeddings. "
            "Run: pip install langchain-huggingface sentence-transformers"
        )

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


def build_faiss_index(
    chunks: List[Document],
    embedding_model: Any,
    save_path: Optional[str] = None
) -> Any:
    """
    Generates vector embeddings for all chunks and indexes them into a FAISS vector store.
    Optionally saves the index to disk for persistence.

    Args:
        chunks: List of semantic text chunk Documents.
        embedding_model: Initialized embedding model instance.
        save_path: Optional folder directory to persist the FAISS index.

    Returns:
        FAISS vectorstore instance.
    """
    if not chunks:
        raise ValueError("Cannot build index: chunk list is empty.")
    if FAISS is None:
        raise ImportError("FAISS is required. Run: pip install faiss-cpu")

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)

    return vectorstore


def load_faiss_index(load_path: str, embedding_model: Any) -> Any:
    """
    Loads a persisted FAISS vector index from disk.

    Args:
        load_path: Folder containing index.faiss and index.pkl.
        embedding_model: Embedding model with identical dimensions.

    Returns:
        Loaded FAISS vectorstore instance.
    """
    if FAISS is None:
        raise ImportError("FAISS is required. Run: pip install faiss-cpu")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"FAISS index folder not found at: {load_path}")

    return FAISS.load_local(load_path, embedding_model, allow_dangerous_deserialization=True)


def process_pdf_documents(
    file_sources: List[Union[str, io.BytesIO, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    save_dir: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], List[Document]]:
    """
    End-to-end ingestion pipeline:
    1. Extracts text and page metadata from all provided PDF sources.
    2. Performs recursive semantic chunking.
    3. Computes vector embeddings and builds the FAISS index.
    4. Returns vectorstore, summary statistics, and chunks.

    Returns:
        (vectorstore, stats_dict, chunk_list)
    """
    start_time = time.time()
    all_pages: List[Document] = []

    for src in file_sources:
        pages = extract_text_from_pdf(src)
        all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No readable text could be extracted from the uploaded PDF file(s).")

    chunks = split_documents(all_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model = get_embedding_model(embedding_model_name)
    vectorstore = build_faiss_index(chunks, embedding_model, save_path=save_dir)

    elapsed_time = round(time.time() - start_time, 2)

    stats = {
        "total_documents": len(file_sources),
        "total_pages": len(all_pages),
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model_name,
        "processing_time_sec": elapsed_time
    }

    return vectorstore, stats, chunks

