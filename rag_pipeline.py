def run_rag_pipeline(query):
    # placeholder for actual RAG pipeline
    retrieved_docs = [
        "Adults should consume around 0.8g protein per kg body weight."
    ]

    answer = "You should eat 120g protein per day."

    return {
        "query": query,
        "answer": answer,
        "contexts": retrieved_docs,
        "retrieved_docs": retrieved_docs
    }