from evaluation.guideline_matching.model import RecommendationType, Compatibility, COMPATIBILITY_MATRIX

def get_compatibility(a: RecommendationType, b: RecommendationType) -> Compatibility:
    return COMPATIBILITY_MATRIX.get(
        (a, b),
        COMPATIBILITY_MATRIX.get((b, a), Compatibility.PARTIAL)
    )

def compare(rag_recs: list, guideline_recs: list):
    # index guidelines by (nutrient, population_group)
    guide_index = {}
    for g in guideline_recs:
        key = (g.nutrient, g.population_group)
        guide_index.setdefault(key, []).append(g)

    results = []
    for r in rag_recs:
        key = (r.nutrient, r.population_group)
        matches = guide_index.get(key, [])
        if not matches:
            results.append({"rag": r, "status": "no_guideline"})
        else:
            for m in matches:
                results.append({
                    "rag": r,
                    "guideline": m,
                    "status": get_compatibility(r, m)
                })
    return results