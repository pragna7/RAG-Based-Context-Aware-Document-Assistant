---

# 📚 RAG-Based-Context-Aware-Document-Assistant

A **Retrieval-Augmented Generation (RAG)** based AI system that allows users to upload PDF documents and interact with them using natural language questions powered by **Google Gemini AI + FAISS vector search**.

---

## 🚀 Overview

This project combines:

* 🔍 **Semantic Search (Retrieval)** using FAISS
* 🤖 **Large Language Model (Generation)** using Google Gemini
* 📄 PDF parsing and chunking
* 🧠 Embedding-based context understanding

It enables users to upload PDFs and ask questions like a chatbot, receiving accurate, context-aware answers directly from the document.

---

## ✨ Key Features

* 📄 Upload and process PDF documents dynamically
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* 🔍 Semantic search using FAISS vector database
* 🤖 Google Gemini-powered response generation
* ✂️ Smart text chunking with overlap preservation
* 📊 Source context visibility for transparency
* 💬 Interactive Streamlit UI
* ⚡ Fast embedding generation using Sentence Transformers

---

## 🏗️ System Architecture

```
PDF Document
   ↓
Text Extraction (PyPDF2)
   ↓
Chunking (Overlapping Segments)
   ↓
Embedding Generation (MiniLM)
   ↓
FAISS Vector Store
   ↓
User Query → Embedding
   ↓
Similarity Search (Top-K Retrieval)
   ↓
Context Augmentation
   ↓
Google Gemini LLM
   ↓
Final Answer + Source Context
```

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini (1.5 Flash / Pro)
* **Vector DB:** FAISS
* **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
* **PDF Processing:** PyPDF2
* **Language:** Python

---

## 📁 Project Structure

```
RAG-Based-Context-Aware-Document-Assistant/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/pragna7/RAG-Based-Context-Aware-Document-Assistant.git
cd RAG-Based-Context-Aware-Document-Assistant
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🔑 API Configuration

This project uses **Google Gemini API**.

👉 Get API Key:
[https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

Then paste it inside the Streamlit sidebar.

---

## 💡 How It Works

1. Upload a PDF file
2. Text is extracted and split into chunks
3. Each chunk is converted into embeddings
4. Stored inside FAISS vector database
5. User query is embedded
6. Most relevant chunks are retrieved
7. Gemini generates final answer using context

---

## 🧪 Example Questions

* Summarize this document
* What are the key findings?
* What statistics are mentioned?
* Explain the main conclusions
* Give insights from this PDF

---

## 📈 Why This Project is Strong (Recruiter View)

This project demonstrates:

* Real-world **RAG system design**
* Vector database integration (FAISS)
* LLM orchestration (Gemini API)
* Embedding-based semantic search
* End-to-end AI application development
* Production-style architecture thinking

---

## 🚧 Limitations

* Works best with text-based PDFs
* Requires internet for Gemini API
* Large PDFs may need tuning for chunk size

---

## 🔮 Future Improvements

* Multi-PDF chat system
* Chat memory (conversation history)
* OCR support for scanned PDFs
* Cloud deployment (Streamlit/AWS)
* Highlight answers directly in PDF

---

## 👩‍💻 Author

**Pragna Seetha**
GitHub: [https://github.com/pragna7](https://github.com/pragna7)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub to support development.

---


