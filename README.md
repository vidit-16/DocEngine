# 📄 Document Q&A Engine (RAG-based)

A simple AI-powered system that allows users to upload a PDF and ask questions about its content.

Built using a Retrieval-Augmented Generation (RAG) pipeline combining semantic search and LLM-based reasoning.

---

## 🚀 Live Demo

👉 https://doc--engine.streamlit.app/

---

## 🧠 How it Works

1. PDF Processing  
   Extracts text using pdfplumber  

2. Chunking  
   Splits text into smaller segments  

3. Embeddings  
   Converts text into vectors using Sentence Transformers  

4. Vector Search  
   Uses FAISS to retrieve relevant chunks  

5. Answer Generation  
   Uses an LLM (OpenAI API) to generate answers from context  

---

## ⚙️ Tech Stack

- Streamlit  
- pdfplumber  
- SentenceTransformers  
- FAISS  
- OpenAI API  
- Streamlit Cloud  

---

## ✨ Features

- Upload and query any PDF  
- Semantic search using embeddings  
- Context-aware answers  
- Clean UI  
- Optional source snippets  

---

## ⚠️ Limitations

- Retrieval depends on chunking quality  
- Sources are approximate (semantic matches, not exact lines)  
- Large PDFs may slow performance  

---

## 🚀 Future Improvements

- Better retrieval (reranking)  
- Page-level source tracking  
- Multi-document support  
- Conversational memory  

---

## 🧪 Local Setup

git clone https://github.com/vidit-16/doc-engine.git  
cd doc-engine  
pip install -r requirements.txt  

Set your API key:

setx OPENAI_API_KEY "your_api_key"

Run:

streamlit run app.py  

---

## 🧠 Key Concepts

- Retrieval-Augmented Generation (RAG)  
- Semantic Search  
- Vector Similarity (FAISS)  
- LLM-based answer generation  

---

## 👤 Author

Vidit Choudhary
