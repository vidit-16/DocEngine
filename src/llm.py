import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, context):
    prompt = f"""
You are an expert assistant.

Answer the question using ONLY the given context.

RULES:
- Do NOT copy sentences directly
- Give clean structured output
- Use bullet points when possible
- Ignore broken or irrelevant text
- If unclear, say: Not clearly found in document

Context:
{context}

Question:
{query}

Final Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()