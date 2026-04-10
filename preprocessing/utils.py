import os
from pathlib import Path
from typing import List, cast
import json

import fitz
import trafilatura
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
OUTPUT_DIR = "output"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNKS_PATH = str(Path(__file__).resolve().parent.parent / "richtlinien" / "all_chunks.json")
FAISS_INDEX_DIR = str(Path(__file__).resolve().parent.parent / "richtlinien" / "faiss_index")


## ---------------------- For HTML ------------------------

def clean_webfile(html):
    text = trafilatura.extract(html, include_comments=False, include_tables=False)

    if text and len(text.strip()) >= 100:
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")

    # If the file is a browser "view-source" capture, the real page HTML may be inside a <pre>/<xmp> block.
    for pre_tag in soup.find_all(["pre", "xmp", "textarea"]):
        source = pre_tag.get_text()
        if source and "<!doctype" in source.lower() and "<html" in source.lower():
            nested = BeautifulSoup(source, "html.parser")
            for tag_name in ["script", "style", "noscript", "iframe", "header", "footer", "nav", "form", "button", "svg", "meta", "link", "input"]:
                for tag in nested.find_all(tag_name):
                    tag.decompose()
            content = nested.get_text(" ", strip=True)
            if content and len(content) >= 100:
                return content.strip()

    for tag_name in ["script", "style", "noscript", "iframe", "header", "footer", "nav", "form", "button", "svg", "meta", "link", "input"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    content = None
    for candidate in ["article", "main", "body"]:
        node = soup.find(candidate)
        if node:
            content = node.get_text(" ", strip=True)
            if content and len(content) >= 100:
                break

    if not content:
        content = soup.get_text(" ", strip=True)

    return content or ""


## ---------------------- For PDFs ------------------------

def extract_base_text(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = [cast(str, page.get_text("text")) for page in doc]
    return pages


def filter_low_content_pages(pages: List[str], min_alpha_ratio=0.5) -> List[str]:
    filtered = []
    for page in pages:
        text = page.strip()
        if not text:
            continue
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / len(text) >= min_alpha_ratio:
            filtered.append(page)
    return filtered


def remove_repeated_lines(pages: List[str]) -> List[str]:
    import re
    from collections import Counter

    normalized_pages = []
    for p in pages:
        lines = []
        for line in p.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            clean_line = re.sub(r'\s+', ' ', clean_line)
            lines.append(clean_line)
        normalized_pages.append(lines)

    freq = Counter(line for page in normalized_pages for line in page)
    cleaned_pages = []
    for page_lines in normalized_pages:
        cleaned_lines = [line for line in page_lines if freq[line] < 3]
        cleaned_pages.append("\n".join(cleaned_lines))

    return cleaned_pages


def basic_clean(text):
    import html as _html
    import re
    import unicodedata

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _html.unescape(text)

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_table_texts(pdf_path: str) -> List[str]:
    from camelot.io import read_pdf

    tables = read_pdf(pdf_path, pages="all", flavor="lattice")
    table_texts: List[str] = []

    for table in tables:
        df = table.df
        if df.empty:
            continue

        # Convert each table to markdown format
        lines = []
        # Header row
        headers = [str(cell).strip() for cell in df.iloc[0]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # Data rows
        for i in range(1, len(df)):
            row = [str(cell).strip() for cell in df.iloc[i]]
            lines.append("| " + " | ".join(row) + " |")

        md_table = "\n".join(lines)
        if md_table.strip():
            table_texts.append(md_table)

    return table_texts


## ---------------------- For all ------------------------

def build_metadata(doc_path, doc_type):
    root_dir = Path(__file__).resolve().parent.parent
    folder_path = Path(doc_path).resolve().parent.relative_to(root_dir)

    return {
        "folder": str(folder_path).replace("\\", "/"),
        "doc_type": doc_type,
    }


def chunk_text(text, metadata):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    doc = Document(text=text, metadata=metadata)
    nodes = splitter.get_nodes_from_documents([doc])

    return nodes

def build_retriever(chunks_path=CHUNKS_PATH, index_dir=FAISS_INDEX_DIR, k=3):
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    if os.path.exists(index_dir):
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vectorstore.save_local(index_dir)

    return vectorstore.as_retriever(search_kwargs={"k": k})