import re
from pathlib import Path
from typing import List, cast

import fitz
import trafilatura
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.utils import get_tokenizer

DATA_DIR = "data"
OUTPUT_DIR = "output"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 70


## ---------------------- For HTML ------------------------

def _strip_boilerplate(soup):
    import re

    # Structural boilerplate: never carries article content.
    for tag_name in [
        "script", "style", "noscript", "iframe", "header", "footer", "nav",
        "form", "button", "svg", "meta", "link", "input", "aside", "menu", "dialog",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Class/id-based boilerplate that survives favor_recall=True (cookie bars,
    # breadcrumbs, "related articles", share widgets, sidebars, ...).
    junk = re.compile(
        r"(cookie|consent|banner|breadcrumb|sidebar|subscribe|newsletter|"
        r"social|share|advert|promo|related|menu|navigation|skip-link)",
        re.I,
    )
    for attr in ("class", "id"):
        for tag in soup.find_all(attrs={attr: junk}):
            tag.decompose()


def _bulletize(soup):
    # Preserve list/definition structure as markdown bullets BEFORE flattening
    # to text, so the actual list content (the answers) is not lost.
    for item in soup.find_all(["li", "dt", "dd"]):
        item.insert_before("\n- ")
    for br in soup.find_all("br"):
        br.replace_with("\n")


def clean_webfile(html):
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,      # food lists are often rendered as <table>
        include_formatting=True,  # keep "- " list markers
        favor_recall=True,        # be generous: stop pruning borderline list blocks
    )

    if text and len(text.strip()) >= 100:
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")

    # If the file is a browser "view-source" capture, the real page HTML may be inside a <pre>/<xmp> block.
    for pre_tag in soup.find_all(["pre", "xmp", "textarea"]):
        source = pre_tag.get_text()
        if source and "<!doctype" in source.lower() and "<html" in source.lower():
            nested = BeautifulSoup(source, "html.parser")
            _strip_boilerplate(nested)
            _bulletize(nested)
            content = nested.get_text("\n")
            if content and len(content.strip()) >= 100:
                return content.strip()

    _strip_boilerplate(soup)
    _bulletize(soup)

    content = None
    for candidate in ["article", "main", "body"]:
        node = soup.find(candidate)
        if node:
            content = node.get_text("\n")
            if content and len(content.strip()) >= 100:
                break

    if not content:
        content = soup.get_text("\n")

    return (content or "").strip()


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

    # Resolve HTML entities that survived extraction (&amp; -> &).
    text = _html.unescape(text)

    # Drop URLs.
    text = re.sub(r'http\S+', '', text)

    # Normalize unicode forms.
    text = unicodedata.normalize("NFKC", text)

    # De-hyphenate words split across a line break ("inflamma-\ntion" -> "inflammation").
    # Only when the continuation starts lowercase, so compounds like "n-3" stay intact.
    text = re.sub(r'(\w)-\n[ \t]*([a-zäöüß])', r'\1\2', text)

    # Collapse spaces/tabs but PRESERVE line breaks, so sentence/paragraph
    # structure survives for the splitter and list bullets are not flattened.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

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


## ------------------ For reference tables ------------------

# The DGE/ÖGE reference table is a faceted grid (one section per life-stage,
# rows of Nährstoff | Geschlecht | Wert | Einheit | Anmerkung | Typ | Fußnoten).
# Generic markdown extraction shreds it (header detached from rows, numbers split).
# Instead we parse it row-by-row into self-contained natural-language statements,
# one chunk per life-stage section, so every value keeps its age group + context.

def parse_dge_reference(pdf_path: str):
    """Return sections: ordered list of (age_group, [row_cells, ...])."""
    import re

    age_header = re.compile(
        r'^(?:\d+\s*bis unter\s*\d+\s*(?:Monate|Jahre)'
        r'|\d+\s*Jahre und älter'
        r'|\d+\.\s*Trimester'
        r'|Stillende)\s*$'
    )

    def norm(cell):
        return (cell or "").replace("\n", " ").strip()

    doc = fitz.open(pdf_path)
    sections = []   # ordered [(age_group, rows)]
    index = {}      # age_group -> rows (same list object stored in `sections`)
    current = None

    for page in doc:
        page_text = cast(str, page.get_text("text"))
        for line in page_text.splitlines():
            stripped = line.strip()
            if age_header.match(stripped):
                current = stripped
                if current not in index:
                    index[current] = []
                    sections.append((current, index[current]))
                break

        finder = page.find_tables()
        for table in (finder.tables if finder else []):
            rows = table.extract()
            if not rows:
                continue

            # Skip the footnote glossary table, matched by its exact first header
            # cell ("Fußnote"). A substring test on the whole header misfires,
            # because every DATA table also has a column literally named "Fußnoten"
            # -> all data pages would be misread as footnote tables and no reference
            # rows would ever be parsed.
            if norm(rows[0][0]).lower() == "fußnote":
                continue

            if current is None:
                continue
            for row in rows[1:]:
                cells = [norm(c) for c in row]
                # Geschlecht column is the reliable marker of a real data row.
                if len(cells) >= 6 and cells[1] in ("Männlich", "Weiblich"):
                    index[current].append(cells)

    return sections


_REF_SEX = {"Männlich": "Männer", "Weiblich": "Frauen"}


def _ref_row_sentence(age, cells):
    # Anmerkung (cells[4]) and footnote refs (cells[6]) are intentionally dropped.
    nutrient = cells[0]
    sex = _REF_SEX.get(cells[1], cells[1])
    value = cells[2]
    unit = cells[3] if len(cells) > 3 else ""
    rtype = cells[5] if len(cells) > 5 else ""

    if value in ("", "-", None):
        return f"{nutrient} ({age}, {sex}): kein Referenzwert für diese Altersgruppe."

    sentence = f"{nutrient} ({age}, {sex}): {value} {unit}".rstrip()
    if rtype:
        sentence += f" ({rtype})"
    return sentence + "."


def _pack_rows(header, sentences, budget, ntok):
    """Greedily pack whole row-sentences into groups that fit `budget` tokens.

    Each group is re-prefixed with `header`, so every chunk is self-contained. A
    row is never split: a group only flushes BEFORE a row that would overflow.
    """
    groups = []
    cur, cur_tokens = [], ntok(header)
    for sent in sentences:
        line = "- " + sent
        t = ntok("\n" + line)
        if cur and cur_tokens + t > budget:
            groups.append(cur)
            cur, cur_tokens = [], ntok(header)
        cur.append(line)
        cur_tokens += t
    if cur:
        groups.append(cur)
    return ["\n".join([header] + g) for g in groups]


def build_reference_nodes(pdf_path: str, metadata: dict) -> List[TextNode]:
    """DGE table -> self-contained chunks, each capped at CHUNK_SIZE tokens.

    A life-stage section is split into as many chunks as needed (whole rows only);
    every chunk repeats the "Altersgruppe" header so it stands alone.
    """
    sections = parse_dge_reference(pdf_path)
    tokenizer = get_tokenizer()

    def ntok(s):
        return len(tokenizer(s))

    nodes: List[TextNode] = []
    for age_group, rows in sections:
        if not rows:
            continue
        header = f"DGE/ÖGE-Referenzwerte für die Nährstoffzufuhr — Altersgruppe: {age_group}."
        sentences = [_ref_row_sentence(age_group, cells) for cells in rows]
        chunks = _pack_rows(header, sentences, CHUNK_SIZE, ntok)

        for i, text in enumerate(chunks, 1):
            node_metadata = dict(metadata)
            node_metadata["age_group"] = age_group
            if len(chunks) > 1:
                node_metadata["part"] = i
                node_metadata["n_parts"] = len(chunks)
            nodes.append(TextNode(text=text, metadata=node_metadata))
    return nodes


## --------- IOM / National Academies DRI table (US_intstitute_of_Medicine.pdf) ---------

# 6 DRI tables, each spanning two facing pages (row labels only on the LEFT page,
# continuation nutrients on the RIGHT page with no labels). Layout is whitespace-only
# (no ruling lines, so find_tables() finds nothing), text mojibake, footnote letters
# glued onto values, and "*" marks an Adequate Intake.
#
# Strategy: rebuild rows from get_text("words") — cluster words by their y-coordinate
# into rows and order each row left-to-right by x. This gives intact numbers AND true
# row-major order regardless of PyMuPDF's reading-order heuristic (which is column-major
# on some of these pages) and never splits a number (unlike find_tables strategy="text").
# Column order per page is hardcoded (value-verified against the PDF). Only rows whose
# value count == the column count are kept: this drops the sparse EAR infant rows and
# auto-aligns the labelled left page with the unlabelled right page by row order.

_IOM_CATEGORIES = ("Infants", "Children", "Males", "Females", "Pregnancy", "Lactation")
_IOM_AGE_UNITS = ("y", "mo")
_IOM_RANGE_RE = re.compile(r'^\d+[–-]\d+$')

# Latin-1/UTF-8 mojibake repair (longer sequences first). Dashes matter for age parsing.
_IOM_MOJIBAKE = [
    ("â€“", "–"), ("â€”", "—"), ("â", "–"),
    ("Âµ", "µ"), ("Î¼", "µ"), ("Î²", "β"), ("Î±", "α"),
    ("Â­", ""), ("Â", ""),
]


def _iom_fix(s: str) -> str:
    for a, b in _IOM_MOJIBAKE:
        s = s.replace(a, b)
    # Normalise every dash/minus variant to one en-dash: the age ranges use U+2013 on
    # some pages but U+2212 (minus sign) on the UL pages, which broke the range regex.
    s = re.sub(r'[‐‑‒–—―−]', '–', s)
    return s


def _iom_clean_value(token: str) -> str:
    token = (token or "").strip()
    token = re.sub(r'(?<=[\d*])[a-z]$', '', token)   # 25f -> 25, 200*a -> 200*, 1,000g -> 1,000
    token = re.sub(r'^(ND)[a-z]$', r'\1', token)     # NDf -> ND
    return token


def _iom_is_value(token: str) -> bool:
    token = (token or "").strip()
    return token in ("ND", "ND*") or bool(re.match(r'^\d[\d.,]*\*?$', token))


# Each spread: page indices are 0-based. mode "fixed" => single allowance kind;
# mode "rda_ai" => per-value RDA unless it ends with "*" (= Adequate Intake).
IOM_TABLES = [
    {
        "type": "EAR", "mode": "fixed",
        "label": "Estimated Average Requirements (EAR)",
        "left_page": 0, "right_page": 1,
        "left_cols": [("Calcium", "mg/d"), ("Carbohydrate", "g/d"), ("Protein", "g/kg/d"),
                      ("Vitamin A", "µg/d"), ("Vitamin C", "mg/d"), ("Vitamin D", "µg/d"),
                      ("Vitamin E", "mg/d"), ("Thiamin", "mg/d"), ("Riboflavin", "mg/d"),
                      ("Niacin", "mg/d")],
        "right_cols": [("Vitamin B6", "mg/d"), ("Folate", "µg/d"), ("Vitamin B12", "µg/d"),
                       ("Copper", "µg/d"), ("Iodine", "µg/d"), ("Iron", "mg/d"),
                       ("Magnesium", "mg/d"), ("Molybdenum", "µg/d"), ("Phosphorus", "mg/d"),
                       ("Selenium", "µg/d"), ("Zinc", "mg/d")],
    },
    {
        "type": "RDA_AI", "mode": "rda_ai",
        "label": "Recommended Dietary Allowances (RDA) and Adequate Intakes (AI), Vitamins",
        "left_page": 2, "right_page": 3,
        "left_cols": [("Vitamin A", "µg/d"), ("Vitamin C", "mg/d"), ("Vitamin D", "µg/d"),
                      ("Vitamin E", "mg/d"), ("Vitamin K", "µg/d"), ("Thiamin", "mg/d")],
        "right_cols": [("Riboflavin", "mg/d"), ("Niacin", "mg/d"), ("Vitamin B6", "mg/d"),
                       ("Folate", "µg/d"), ("Vitamin B12", "µg/d"), ("Pantothenic Acid", "mg/d"),
                       ("Biotin", "µg/d"), ("Choline", "mg/d")],
    },
    {
        "type": "RDA_AI", "mode": "rda_ai",
        "label": "Recommended Dietary Allowances (RDA) and Adequate Intakes (AI), Elements",
        "left_page": 4, "right_page": 5,
        "left_cols": [("Calcium", "mg/d"), ("Chromium", "µg/d"), ("Copper", "µg/d"),
                      ("Fluoride", "mg/d"), ("Iodine", "µg/d"), ("Iron", "mg/d"),
                      ("Magnesium", "mg/d")],
        "right_cols": [("Manganese", "mg/d"), ("Molybdenum", "µg/d"), ("Phosphorus", "mg/d"),
                       ("Selenium", "µg/d"), ("Zinc", "mg/d"), ("Potassium", "mg/d"),
                       ("Sodium", "mg/d"), ("Chloride", "g/d")],
    },
    {
        "type": "RDA_AI", "mode": "rda_ai",
        "label": "Recommended Dietary Allowances (RDA) and Adequate Intakes (AI), "
                 "Total Water and Macronutrients",
        "left_page": 6, "right_page": None,
        "left_cols": [("Total Water", "L/d"), ("Carbohydrate", "g/d"), ("Total Fiber", "g/d"),
                      ("Fat", "g/d"), ("Linoleic Acid", "g/d"), ("α-Linolenic Acid", "g/d"),
                      ("Protein", "g/d")],
        "right_cols": [],
    },
    {
        "type": "UL", "mode": "fixed",
        "label": "Tolerable Upper Intake Levels (UL), Vitamins",
        "left_page": 7, "right_page": 8,
        "left_cols": [("Vitamin A", "µg/d"), ("Vitamin C", "mg/d"), ("Vitamin D", "µg/d"),
                      ("Vitamin E", "mg/d"), ("Vitamin K", ""), ("Thiamin", ""),
                      ("Riboflavin", "")],
        "right_cols": [("Niacin", "mg/d"), ("Vitamin B6", "mg/d"), ("Folate", "µg/d"),
                       ("Vitamin B12", ""), ("Pantothenic Acid", ""), ("Biotin", ""),
                       ("Choline", "g/d"), ("Carotenoids", "")],
    },
    {
        "type": "UL", "mode": "fixed",
        "label": "Tolerable Upper Intake Levels (UL), Elements",
        "left_page": 9, "right_page": 10,
        "left_cols": [("Arsenic", ""), ("Boron", "mg/d"), ("Calcium", "mg/d"),
                      ("Chromium", ""), ("Copper", "µg/d"), ("Fluoride", "mg/d"),
                      ("Iodine", "µg/d"), ("Iron", "mg/d"), ("Magnesium", "mg/d"),
                      ("Manganese", "mg/d")],
        "right_cols": [("Molybdenum", "µg/d"), ("Nickel", "mg/d"), ("Phosphorus", "g/d"),
                       ("Potassium", ""), ("Selenium", "µg/d"), ("Silicon", ""),
                       ("Sulfate", ""), ("Vanadium", "mg/d"), ("Zinc", "mg/d"),
                       ("Sodium", ""), ("Chloride", "g/d")],
    },
]


def _iom_word_rows(page, ytol: float = 4.0) -> List[List[str]]:
    """Cluster get_text('words') by y into rows; order each row left-to-right by x.

    The row anchor (cy) is the y of the row's FIRST word and stays fixed until a new
    row starts — a drifting anchor chains wide-spaced rows together on the UL pages.
    """
    words = [(w[0], w[1], _iom_fix(w[4])) for w in page.get_text("words")]
    rows, cur, cy = [], [], None
    for x0, y0, txt in sorted(words, key=lambda w: w[1]):
        if cy is None or abs(y0 - cy) <= ytol:
            cur.append((x0, txt))
            if cy is None:
                cy = y0
        else:
            rows.append([t for _, t in sorted(cur)])
            cur = [(x0, txt)]
            cy = y0
    if cur:
        rows.append([t for _, t in sorted(cur)])
    return rows


def _iom_split_age(row):
    """Return (age_label, value_tokens) if the row starts with a life-stage age, else None."""
    if row[0] == ">" and len(row) >= 3 and row[2] in _IOM_AGE_UNITS:
        return f"> {row[1]} {row[2]}", row[3:]
    if _IOM_RANGE_RE.match(row[0]) and len(row) >= 2 and row[1] in _IOM_AGE_UNITS:
        return f"{row[0]} {row[1]}", row[2:]
    return None, None


def _iom_left_rows(page, n_cols):
    """Ordered fully-populated data rows from a labelled page: (category, age, [values])."""
    out, category = [], None
    for row in _iom_word_rows(page):
        if not row:
            continue
        if row[0].startswith("NOTES") or row[0].startswith("SOURCES"):
            break
        if len(row) == 1 and row[0] in _IOM_CATEGORIES:
            category = row[0]
            continue
        age, value_tokens = _iom_split_age(row)
        if age is None:
            continue
        values = [_iom_clean_value(t) for t in value_tokens]
        if len(values) == n_cols and all(_iom_is_value(v) for v in values):
            out.append((category, age, values))
    return out


def _iom_right_rows(page, n_cols):
    """Ordered fully-populated value rows from an unlabelled continuation page."""
    out = []
    for row in _iom_word_rows(page):
        if not row:
            continue
        if row[0].startswith("NOTES") or row[0].startswith("SOURCES"):
            break
        values = [_iom_clean_value(t) for t in row]
        if len(values) == n_cols and all(_iom_is_value(v) for v in values):
            out.append(values)
    return out


def _iom_row_text(spec, category, age, values):
    cols = spec["left_cols"] + spec["right_cols"]
    header = (f"Dietary Reference Intakes (DRIs) — {spec['label']}. "
              f"Life-stage group: {category}, {age}.")
    body = []
    for (nutrient, unit), value in zip(cols, values):
        value = value.strip()
        if not value or value.upper().startswith("ND"):
            continue
        if spec["mode"] == "rda_ai":
            kind = "AI" if value.endswith("*") else "RDA"
            value = value.rstrip("*")
            body.append(f"- {nutrient}: {value} {unit} ({kind}).".replace("  ", " "))
        else:
            body.append(f"- {nutrient}: {value} {unit}.".replace("  ", " "))
    if not body:
        return None
    return "\n".join([header] + body)


def build_iom_nodes(pdf_path: str, metadata: dict) -> List[TextNode]:
    """One self-contained chunk per (DRI table, life-stage group)."""
    doc = fitz.open(pdf_path)
    nodes: List[TextNode] = []

    for spec in IOM_TABLES:
        left = _iom_left_rows(doc[spec["left_page"]], len(spec["left_cols"]))

        if spec["right_page"] is None:
            rows = [(c, age, vals) for (c, age, vals) in left]
        else:
            right = _iom_right_rows(doc[spec["right_page"]], len(spec["right_cols"]))
            if len(left) != len(right):
                print(f"  IOM warning [{spec['label']}]: {len(left)} left vs "
                      f"{len(right)} right full rows — aligning by order, verify output.")
            rows = [(c, age, lvals + rvals) for (c, age, lvals), rvals in zip(left, right)]

        for category, age, values in rows:
            text = _iom_row_text(spec, category, age, values)
            if text is None:
                continue
            node_metadata = dict(metadata)
            node_metadata["table_type"] = spec["type"]
            node_metadata["life_stage"] = f"{category} {age}".strip()
            nodes.append(TextNode(text=text, metadata=node_metadata))

    doc.close()
    return nodes


## ---------------------- For all ------------------------

def detect_lang(text):
    """Best-effort language tag in {'en', 'de'} for a chunk.

    Used only for per-language analysis in evaluation (slicing metrics by
    document language), not for retrieval. Restricted to the two languages
    present in the corpus. Falls back to a German-cue heuristic if py3langid
    isn't installed."""
    sample = (text or "").strip()
    if not sample:
        return "unknown"
    try:
        import py3langid

        py3langid.set_languages(["en", "de"])
        return py3langid.classify(sample)[0]
    except Exception:
        lowered = f" {sample.lower()} "
        de_cues = (
            "ä", "ö", "ü", "ß", " der ", " die ", " und ", " für "
        )
        return "de" if any(cue in lowered for cue in de_cues) else "en"


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
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )

    doc = Document(text=text, metadata=metadata)
    nodes = splitter.get_nodes_from_documents([doc])

    return nodes

