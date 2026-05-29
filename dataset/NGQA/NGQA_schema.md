{
  "id": ...,
  "query": ...,
  "context_variants": {
    "food_facts": ...,
    "user_profile": ...,
    "retrieved": ...
  },
  "gold": {
    "is_healthy": ...,
    "conflicts": [
      {
        "condition": ...,
        "nutrient_tag": ...,
        "nutrient_readable": ...
      }
    ],
    "reference_answer": ...,
    "csv_short_answer": ...,
    "is_healthy_agrees_with_csv_answer": ...
  },
  "difficulty": ...,
  "knowledge_required": {
    "food_facts": [...],
    "clinical_rules": [
      {
        "rule": ...,
        "condition": ...,
        "nutrient_tag": ...
      }
    ]
  },
  "raw_graph": {
    "nodes": [
      [
        ...,
        {
          "name": ...,
          "attr": ...
        }
      ]
    ],
    "edges": [
      [
        ...,
        ...,
        ...
      ]
    ]
  }
}