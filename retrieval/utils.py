import os
from typing import List, cast

import fitz
from camelot.io import read_pdf
import trafilatura
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

DATA_DIR = "data"
OUTPUT_DIR = "output"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


## ---------------------- For HTML ------------------------

def clean_webfile(html):
    text = trafilatura.extract(html)

    if not text or len(text) < 100:
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        if article:
            text = article.get_text(" ", strip=True)


## ---------------------- For PDFs ------------------------

def extract_base_text(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = [cast(str, page.get_text("text")) for page in doc]
    return pages


def remove_repeated_lines(pages: List[str]) -> List[str]:
    from collections import Counter

    lines = [line for p in pages for line in p.split("\n")]
    freq = Counter(lines)

    cleaned_pages = []
    for p in pages:
        cleaned_lines = [l for l in p.split("\n") if freq[l] < 3]
        cleaned_pages.append("\n".join(cleaned_lines))

    return cleaned_pages


def basic_clean(text):
    import re
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_table_texts(pdf_path: str) -> List[str]:
    tables = read_pdf(pdf_path, pages="all", flavor="lattice")
    table_texts: List[str] = []

    for table in tables:
        df = table.df
        headers = df.iloc[0]  # assume first row = header
        lines = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            parts = [
                f"{headers[j]}: {row[j]}"
                for j in range(len(headers))
                if row[j]
            ]
            lines.append(", ".join(parts))

        table_texts.append("\n".join(lines))

    return table_texts


## ---------------------- For all ------------------------

def build_metadata(doc_path, doc_type):
    return {
        "folder": os.path.basename(doc_path),
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