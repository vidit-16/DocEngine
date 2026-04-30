def chunk_text(text, chunk_size=400, overlap=50):
    text = text.replace("\n", " ")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 40:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks