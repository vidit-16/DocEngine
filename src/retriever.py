import numpy as np

def search(query, model, index, chunks, k=5):
    query_lower = query.lower()
    query_words = query_lower.split()

    # 🔥 keyword-first retrieval
    keyword_hits = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in query_words):
            keyword_hits.append(chunk)

    if len(keyword_hits) >= 3:
        return keyword_hits[:k]

    # fallback semantic search
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in indices[0]]