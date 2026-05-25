"""
NGQA data processor for RAG evaluation.

Converts NGQA graph samples (nodes + edges) into evaluation samples with:
- A natural language query
- Verbalized context (food facts + user profile), with NO contradiction info
- A gold answer derived from the 'contradict' edges
- A clean split between food-side facts and clinical rules, for retrieval evaluation
"""

import ast
import csv
import json
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Tag humanization
# ---------------------------------------------------------------------------
TAG_MAP = {
    'low_carb': 'low in carbohydrates',   'high_carb': 'high in carbohydrates',
    'low_sugar': 'low in sugar',          'high_sugar': 'high in sugar',
    'low_sodium': 'low in sodium',        'high_sodium': 'high in sodium',
    'low_protein': 'low in protein',      'high_protein': 'high in protein',
    'low_cholesterol': 'low in cholesterol', 'high_cholesterol': 'high in cholesterol',
    'low_fat': 'low in fat',              'high_fat': 'high in fat',
    'low_fiber': 'low in fiber',          'high_fiber': 'high in fiber',
    'low_calorie': 'low in calories',     'high_calorie': 'high in calories',
}


def humanize_tag(tag: str) -> str:
    """Turn 'high_sodium' into 'high in sodium'. Fall back to underscore->space."""
    return TAG_MAP.get(tag, tag.replace('_', ' '))


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------
def _parse_graph(nodes: list, edges: list) -> dict:
    """
    Parse the raw node/edge lists into a structured dict.

    Identifies the food and user anchor nodes, then groups everything else
    by its relationship to those two anchors.

    Returns a dict with: food_name, category, ingredients, nutrition_tags,
    conditions, habits, contradictions, node_dict.
    """
    node_dict = {n[0]: n[1] for n in nodes}

    # Group outgoing edges by source node for easy lookup
    outgoing = defaultdict(list)
    for src, rel, tgt in edges:
        outgoing[src].append((rel, tgt))

    # The user node has attr['attr'] == 'user' (its 'name' is the NHANES user ID)
    user_id = next(nid for nid, attr in node_dict.items()
                   if attr.get('attr') == 'user')
    # The food node has a numeric 'name' (the FNDDS food code) and isn't the user
    food_id = next(nid for nid, attr in node_dict.items()
                   if isinstance(attr['name'], int) and nid != user_id)

    # --- Food-side information ---
    food_name = node_dict[food_id]['attr']
    category = next(
        (node_dict[t]['attr'] for _, t in outgoing[food_id]
         if node_dict[t]['name'] == 'category'),
        None,
    )
    ingredients = [node_dict[t]['attr'] for _, t in outgoing[food_id]
                   if node_dict[t]['name'] == 'ingredient']
    nutrition_tags = [node_dict[t]['attr'] for _, t in outgoing[food_id]
                      if node_dict[t]['name'] == 'food_nutrition_tag']

    # --- User-side information ---
    conditions = [node_dict[t]['attr'] for _, t in outgoing[user_id]
                  if node_dict[t]['name'] == 'status']
    habits = [node_dict[t]['attr'] for _, t in outgoing[user_id]
              if node_dict[t]['name'] == 'dietary habit']

    # --- Contradictions: the clinical rule edges (gold answer signal) ---
    # These have the form (condition_node, 'contradict', nutrition_tag_node)
    contradictions = []
    for src, rel, tgt in edges:
        if rel == 'contradict':
            contradictions.append({
                'condition': node_dict[src]['attr'],
                'nutrient_tag': node_dict[tgt]['attr'],
            })

    return {
        'food_name': food_name,
        'category': category,
        'ingredients': ingredients,
        'nutrition_tags': nutrition_tags,
        'conditions': conditions,
        'habits': habits,
        'contradictions': contradictions,
        'node_dict': node_dict,
    }


# ---------------------------------------------------------------------------
# Verbalization (graph -> natural language)
# ---------------------------------------------------------------------------
def _verbalize_food(parsed: dict) -> str:
    """Turn the food-side of the graph into a natural-language description."""
    parts = [f"The food is {parsed['food_name']}"]
    if parsed['category']:
        parts.append(f", which belongs to the {parsed['category']} category")
    parts.append(".")
    if parsed['ingredients']:
        parts.append(f" It contains the following ingredients: "
                     f"{', '.join(parsed['ingredients'])}.")
    if parsed['nutrition_tags']:
        readable = [humanize_tag(t) for t in parsed['nutrition_tags']]
        parts.append(f" Nutritionally, it is {', '.join(readable)}.")
    return "".join(parts)


def _verbalize_user(parsed: dict) -> str:
    """Turn the user-side of the graph into a natural-language description."""
    parts = ["The user "]
    if parsed['conditions']:
        parts.append(f"has the following health conditions: "
                     f"{', '.join(parsed['conditions'])}. ")
    if parsed['habits']:
        parts.append(f"Their dietary habits include: "
                     f"{'; '.join(parsed['habits'])}.")
    return "".join(parts).strip()


def _compose_query(question: str, food_facts: str, user_profile: str) -> str:
    """
    Build the self-contained generation input: question + food info + user info.

    This is what a baseline (no-RAG) LLM receives directly, and what RAG
    configurations extend by appending retrieved guideline context.
    """
    return (
        f"Question: {question}\n\n"
        f"Food information:\n{food_facts}\n\n"
        f"User profile:\n{user_profile}"
    )


# ---------------------------------------------------------------------------
# Gold answer construction
# ---------------------------------------------------------------------------
def _build_gold(parsed: dict) -> dict:
    """
    Build the gold answer in three granularities:
      - is_healthy: bool, for binary classification metrics
      - conflicts: structured list, for multi-label metrics (matches NGQA -ML task)
      - summary: natural-language answer, for LLM-as-judge / semantic scoring
    """
    contradictions = parsed['contradictions']

    conflicts = [
        {
            'condition': c['condition'],
            'nutrient_tag': c['nutrient_tag'],
            'nutrient_readable': humanize_tag(c['nutrient_tag']),
        }
        for c in contradictions
    ]

    if not contradictions:
        summary = "This food appears suitable for the user based on the given profile."
    else:
        phrases = [
            f"the user's {c['condition']} conflicts with the food being "
            f"{humanize_tag(c['nutrient_tag'])}"
            for c in contradictions
        ]
        summary = "This food is not recommended because " + "; ".join(phrases) + "."

    return {
        'is_healthy': len(contradictions) == 0,
        'conflicts': conflicts,
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# Knowledge split (food facts vs. clinical rules)
# ---------------------------------------------------------------------------
def _build_knowledge_split(parsed: dict) -> dict:
    """
    Split the knowledge needed to answer the question into:
      - food_facts: food-side facts (probably NOT in your RAG knowledge base).
                    Your system should treat these as given input.
      - clinical_rules: condition -> nutrient conflict rules (SHOULD be in your
                    knowledge base, e.g. clinical guidelines). These are the
                    gold retrieval targets.
    """
    food_facts = [
        f"{parsed['food_name']} is {humanize_tag(t)}"
        for t in parsed['nutrition_tags']
    ]

    clinical_rules = [
        {
            'rule': (f"Patients with {c['condition']} should limit foods "
                     f"that are {humanize_tag(c['nutrient_tag'])}"),
            'condition': c['condition'],
            'nutrient_tag': c['nutrient_tag'],
        }
        for c in parsed['contradictions']
    ]

    return {
        'food_facts': food_facts,
        'clinical_rules': clinical_rules,
    }


# ---------------------------------------------------------------------------
# Main entry point: one graph -> one evaluation sample
# ---------------------------------------------------------------------------

def graph_to_sample(
    nodes: list,
    edges: list,
    question: str,
    question_id: Any = None,
    reference_answer: str | None = None,
    difficulty: str | None = None,
) -> dict:
    """
    Convert a single NGQA graph sample into a RAG evaluation sample.

    Returns a dict with:
      - id: identifier for the sample
      - query: self-contained generation input (query + food_facts + user_profile);
          what a baseline LLM receives, and the base RAG configs extend with retrieved context
      - context_variants:
          * 'food_facts': structured food info (from food DB in production)
          * 'user_profile': structured user info (from user profile in production)
          * 'retrieved': None (to be filled by your RAG system at eval time)
      - gold: multi-granularity ground-truth answer
          * 'reference_answer': the original NGQA answer string, if provided
      - difficulty: difficulty level from the source CSV, if provided
      - knowledge_required: split between food facts and clinical rules;
          clinical_rules are the gold retrieval targets
      - raw_graph: original nodes/edges, for re-processing if needed
    """
    parsed = _parse_graph(nodes, edges)
    gold = _build_gold(parsed)
    if reference_answer is not None:
        gold['reference_answer'] = reference_answer

    food_facts = _verbalize_food(parsed)
    user_profile = _verbalize_user(parsed)

    return {
        'id': question_id,
        'query': _compose_query(question, food_facts, user_profile),
        'context_variants': {
            # Always-provided structured context (not retrieved in a real system)
            'food_facts': food_facts,
            'user_profile': user_profile,
            # Populated later by RAG pipeline during evaluation
            'retrieved': None,
        },
        'gold': gold,
        'difficulty': difficulty,
        'knowledge_required': _build_knowledge_split(parsed),
        'raw_graph': {'nodes': nodes, 'edges': edges},
    }


# ---------------------------------------------------------------------------
# Batch processing: NGQA CSV -> JSONL evaluation file
# ---------------------------------------------------------------------------
def _parse_cell(cell: Any) -> list:
    """
    Parse a node_list or edge_list cell.

    The cells are stored as Python-style strings (single quotes, int keys),
    not JSON, so we use ast.literal_eval. If the cell is already a list
    (e.g. passed in from memory), return it as-is.
    """
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        raise TypeError(f"Expected str or list, got {type(cell).__name__}")
    return ast.literal_eval(cell)


def process_csv(
    input_path: str,
    output_path: str,
    id_column: str | None = None,
) -> int:
    """
    Read an NGQA CSV and write a JSONL file of evaluation samples.
    Only the *hard* question/answer columns are used; easy and medium
    columns are ignored.

    Expected columns:
      - question_hard: the natural-language question
      - answer_hard:   the reference answer
      - node_list:     Python-repr string of the node list
      - edge_list:     Python-repr string of the edge list
      - difficulty:    difficulty level, for filtering or analysis
      - (optional) id_column: whatever column holds a stable sample ID

    Returns the number of samples written.
    """
    required = ['question_hard', 'answer_hard', 'node_list', 'edge_list', 'difficulty']
    count = 0

    with open(input_path, 'r', encoding='utf-8', newline='') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. "
                             f"Found columns: {reader.fieldnames}")

        for i, row in enumerate(reader):
            # Skip rows where any required field is empty
            if any(not (row.get(c) or '').strip() for c in required):
                continue

            sample = graph_to_sample(
                nodes=_parse_cell(row['node_list']),
                edges=_parse_cell(row['edge_list']),
                question_id=i,
                question=row['question_hard'].strip(),
                reference_answer=row['answer_hard'].strip(),
                difficulty=row['difficulty'].strip()
            )
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            count += 1
    return count


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
   
    n = process_csv('NGQA.csv', 'NGQA.jsonl')
    print(f"Processed {n} samples")