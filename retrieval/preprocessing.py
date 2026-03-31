import os
import json
import utils

DATA_DIR = "../richtlinien"
OUTPUT_DIR = "."

def process_html(html_path):
    pages = utils.extract_base_text(html_path)

    text = "\n".join(pages)
    text = utils.clean_webfile(text)
    text = utils.basic_clean(text)

    metadata = utils.build_metadata(html_path, "web")
    return text, metadata

def process_normal_pdf(pdf_path):
    pages = utils.extract_base_text(pdf_path)
    pages = utils.remove_repeated_lines(pages)

    text = "\n".join(pages)
    text = utils.basic_clean(text)

    metadata = utils.build_metadata(pdf_path, "normal")
    return text, metadata


def process_table_pdf(pdf_path):
    table_texts = utils.extract_table_texts(pdf_path)
    text = "\n\n".join(table_texts)

    metadata = utils.build_metadata(pdf_path, "table")
    return text, metadata


def run_preprocessing_pipeline():
    all_nodes = []

    folder_map = {
        "PDF": f"{DATA_DIR}/normal_pdfs",
        "HTML": f"{DATA_DIR}/web_HTML",
        "PDF_table": f"{DATA_DIR}/table_pdfs",
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
                text, metadata = process_table_pdf(path)

            # ---- ADD DOC TYPE TO METADATA ----
            metadata["doc_type"] = doc_type
            metadata["source"] = file

            # ---- CHUNKING ----
            nodes = utils.chunk_text(text, metadata)
            all_nodes.extend(nodes)

    # ---- SAVE ----
    os.makedirs(f"{OUTPUT_DIR}/chunks", exist_ok=True)

    serialized = [
        {
            "text": n.text,
            "metadata": n.metadata
        }
        for n in all_nodes
    ]

    with open(f"{OUTPUT_DIR}/chunks/all_chunks.json", "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"Processed {len(all_nodes)} chunks.")


if __name__ == "__main__":
    run_preprocessing_pipeline()