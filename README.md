
# 📚 RAG-Based-Context-Aware-Document-Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that transforms PDF documents into an intelligent, context-aware question answering assistant using Google Gemini, FAISS, and Transformer-based embeddings.

🚀 Overview

The RAG-Based-Context-Aware-Document-Assistant is a GenAI-powered application that enables users to interact with PDF documents using natural language queries.

It combines semantic search (retrieval) with large language model reasoning (generation) to deliver accurate, context-grounded responses.

Instead of manually reading documents, users can simply upload a PDF and ask questions like a chatbot.

🎯 Key Features
📄 Upload and process PDF documents dynamically
🧠 Retrieval-Augmented Generation (RAG) pipeline implementation
🔍 Semantic search using FAISS vector database
🤖 Google Gemini-powered intelligent response generation
✂️ Smart text chunking with overlap preservation
📊 Source context visibility for transparency
💬 Interactive Streamlit-based UI
⚡ Fast embedding generation using Sentence Transformers
🏗️ System Architecture
PDF Document
     ↓
Text Extraction (PyPDF2)
     ↓
Chunking (Overlapping Segments)
     ↓
Embedding Generation (all-MiniLM-L6-v2)
     ↓
Vector Store (FAISS Index)
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
🛠️ Tech Stack
Layer	Technology
Frontend	Streamlit
LLM	Google Gemini (1.5 Flash / Pro)
Vector Database	FAISS
Embeddings	Sentence Transformers (MiniLM)
PDF Processing	PyPDF2
Language	Python
📂 Project Structure
RAG-Based-Context-Aware-Document-Assistant/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/pragna7/RAG-Based-Context-Aware-Document-Assistant.git
cd RAG-Based-Context-Aware-Document-Assistant
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run app.py
🔑 API Configuration

This project uses Google Gemini API for response generation.

👉 Get API Key:
https://makersuite.google.com/app/apikey

Then enter the API key in the application sidebar.

💡 How It Works (RAG Pipeline)
📄 PDF document is uploaded
✂️ Text is extracted and split into chunks
🔢 Embeddings are generated for each chunk
📦 Stored in FAISS vector index
🔍 User query is converted into embeddings
🎯 Relevant chunks are retrieved using similarity search
🤖 Gemini generates final response using retrieved context
🧪 Example Queries
“Summarize this document”
“What are the key findings?”
“What statistics are mentioned?”
“Explain the main conclusions”
“Give insights from this document”
📈 Engineering Highlights
Efficient chunk-based retrieval strategy
Embedding-driven semantic search (not keyword search)
Reduced hallucination via context grounding
Modular RAG pipeline architecture
Stateless session-based design using Streamlit
🚧 Limitations
Scanned PDFs require OCR (not included)
Large documents may need chunk optimization tuning
Requires internet access for Gemini API
🔮 Future Enhancements
🔥 Multi-document RAG support
🔥 Conversational memory (chat history)
🔥 OCR support for scanned PDFs
🔥 Highlight answers inside PDF viewer
🔥 Cloud deployment (Streamlit / AWS)
🔥 Authentication system for users
👩‍💻 Author

Pragna Seetha
GitHub: @pragna7

⭐ Repository Impact

If you found this project useful, please consider giving a ⭐ to support further development.

🧠 Why this project matters

This project demonstrates:

Real-world GenAI system design
Strong understanding of RAG architecture
Practical use of vector databases (FAISS)
Integration of LLMs with external knowledge sources
End-to-end AI application development skills
