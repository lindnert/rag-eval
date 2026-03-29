def run_rag_pipeline(query):
    # placeholder for actual RAG pipeline
    retrieved_docs = [
        "Adults should consume around 0.8g protein per kg body weight."
    ]

    answer = """Hier ist ein Überblick, was du berücksichtigen solltest, basierend auf deinen Angaben (29 Jahre, 71kg):

1. Grundprinzipien:

Kalorienüberschuss: Um Muskeln aufzubauen, brauchst du mehr Kalorien, als du verbrauchst. Ein Überschuss von 250-500 Kalorien pro Tag ist ein guter Ausgangspunkt. Das lässt dir Spielraum, ohne dass du sofort zu viel Fett ansetzt.
Hohe Proteinzufuhr: Protein ist der Baustein für Muskeln.
Ausreichend Kohlenhydrate: Kohlenhydrate liefern Energie für dein Training und helfen, die Regeneration zu unterstützen.
Gesunde Fette: Gesunde Fette sind wichtig für die Hormonproduktion und die allgemeine Gesundheit.
2. Makronährstoffe – Empfehlungen:

Protein: 1,6 - 2,2 Gramm pro Kilogramm Körpergewicht. Das wären für dich etwa 114 - 156 Gramm Protein pro Tag. Gute Proteinquellen: Hähnchen, Fisch, Rindfleisch, Eier, Milchprodukte, Tofu, Linsen, Bohnen, Proteinshakes.
Kohlenhydrate: 4-5 Gramm pro Kilogramm Körpergewicht. Das wären etwa 284 - 355 Gramm Kohlenhydrate pro Tag. Gute Kohlenhydratquellen: Vollkornprodukte, Reis, Kartoffeln, Obst, Gemüse.
Fette: 0,8 - 1,2 Gramm pro Kilogramm Körpergewicht. Das wären etwa 57 - 85 Gramm Fett pro Tag. Gute Fettquellen: Avocado, Nüsse, Samen, Olivenöl, Fisch.
3. Mikronährstoffe:

Vitamine & Mineralstoffe: Achte auf eine ausgewogene Ernährung mit viel Obst und Gemüse, um alle notwendigen Vitamine und Mineralstoffe zu bekommen. Besonders wichtig sind:
Vitamin D: Wichtig für die Muskelkraft und -funktion.
Magnesium: Spielt eine Rolle bei Muskelkontraktionen und Nervenfunktion.
Zink: Wichtig für das Muskelwachstum und die Wundheilung."""

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
        "retrieved_docs": retrieved_docs
    }