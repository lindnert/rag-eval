import os
import sys
import json
from pathlib import Path

# Running this file directly puts only preprocessing/ on sys.path (so `import utils`
# works) but not the project root, so `retrieval` can't be found. Add the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils
from retrieval import build_retriever

DATA_DIR = str(Path(__file__).resolve().parent.parent / "richtlinien")
OUTPUT_DIR = DATA_DIR

# Tables that need a dedicated row-by-row parser instead of generic markdown extraction.
REFERENCE_TABLE_BUILDERS = {
    "DGE-Referenzwerte.pdf": utils.build_reference_nodes,
    "US_intstitute_of_Medicine.pdf": utils.build_iom_nodes,
}

def process_html(html_path):
    # Read raw bytes so trafilatura/BeautifulSoup can detect the real charset.
    # (read_text(errors="ignore") silently dropped undecodable bytes -> mojibake.)
    html = Path(html_path).read_bytes()
    text = utils.clean_webfile(html)
    text = utils.basic_clean(text)

    metadata = utils.build_metadata(html_path, "web")
    return text, metadata

def process_normal_pdf(pdf_path):
    pages = utils.extract_base_text(pdf_path)
    pages = utils.remove_repeated_lines(pages)
    pages = utils.filter_low_content_pages(pages)

    # Join with a blank line so page breaks become paragraph boundaries
    # (the splitter prefers to break there, not mid-sentence).
    text = "\n\n".join(pages)
    text = utils.basic_clean(text)

    metadata = utils.build_metadata(pdf_path, "normal")
    return text, metadata


def process_table_pdf(pdf_path):
    table_texts = utils.extract_table_texts(pdf_path)
    if not table_texts:
        print(f"  Warning: no tables extracted from {pdf_path}")
        return "", utils.build_metadata(pdf_path, "table")

    text = "\n\n".join(table_texts)

    metadata = utils.build_metadata(pdf_path, "table")
    return text, metadata


def generate_chunks():
    all_nodes = []

    folder_map = {
        "PDF": f"{DATA_DIR}/PDF",
        "HTML": f"{DATA_DIR}/HTML",
        "PDF_table": f"{DATA_DIR}/PDF_table",
    }

    for doc_type, folder in folder_map.items():
        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            # ---- FILE TYPE HANDLING ----
            if doc_type == "PDF":
                if not file.endswith(".pdf"):
                    continue
                text, metadata = process_normal_pdf(path)

            elif doc_type == "HTML":
                if not file.endswith(".html"):
                    continue
                text, metadata = process_html(path)

            elif doc_type == "PDF_table":
                if not file.endswith(".pdf"):
                    continue
                if file in REFERENCE_TABLE_BUILDERS:
                    metadata = utils.build_metadata(path, "table")
                    metadata["doc_type"] = doc_type
                    metadata["source"] = file
                    ref_nodes = REFERENCE_TABLE_BUILDERS[file](path, metadata)
                    if not ref_nodes:
                        print(f"  Warning: no reference rows parsed from {file}")
                    all_nodes.extend(ref_nodes)
                    continue
                text, metadata = process_table_pdf(path)

            metadata["doc_type"] = doc_type
            metadata["source"] = file

            nodes = utils.chunk_text(text, metadata)
            all_nodes.extend(nodes)

    serialized = [
        {
            "text": n.text,
            "metadata": n.metadata
        }
        for n in all_nodes
    ]

    with open(f"{OUTPUT_DIR}/all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_nodes)} chunks.")


if __name__ == "__main__":
    generate_chunks()
    build_retriever()