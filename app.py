import streamlit as st
import PyPDF2
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import io
import re

# Configure page
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_boundary = max(last_period, last_newline)
            
            if last_boundary > start + chunk_size // 2:
                end = start + last_boundary + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if end >= len(text):
            break
    
    return chunks

@st.cache_resource
def load_embeddings_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(text_chunks, embeddings_model):
    """Create FAISS vector store from text chunks"""
    if not text_chunks:
        return None
    
    # Generate embeddings
    embeddings = embeddings_model.encode(text_chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index

def search_similar_chunks(query, vector_store, text_chunks, embeddings_model, k=3):
    """Search for similar chunks using vector similarity"""
    if vector_store is None or not text_chunks:
        return []
    
    # Generate query embedding
    query_embedding = embeddings_model.encode([query])
    
    # Search for similar chunks
    distances, indices = vector_store.search(query_embedding.astype('float32'), k)
    
    # Return relevant chunks
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(text_chunks):
            relevant_chunks.append(text_chunks[idx])
    
    return relevant_chunks

def generate_answer(query, context_chunks, gemini_api_key, model_name="gemini-2.0-flash-exp"):
    """Generate answer using Gemini API with RAG context"""
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    # Use selected Gemini model
    model = genai.GenerativeModel(model_name)
    
    # Prepare context
    context = "\n\n".join(context_chunks)
    
    # Create prompt
    prompt = f"""
    Based on the following context from the PDF document, please answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Please provide a detailed answer based only on the information provided in the context. 
    If the answer cannot be found in the context, please say so.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main application
st.title("📚 PDF RAG Assistant")
st.markdown("Upload a PDF document and ask questions about its content using AI-powered search!")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Select Gemini Model:",
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
        help="Gemini 2.0 Flash is faster, Gemini 1.5 Pro is more capable"
    )
    
    gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
    
    if gemini_api_key:
        st.success("API Key provided ✅")
    else:
        st.warning("Please enter your Gemini API key to continue")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
        
        # Process PDF button
        if st.button("🔄 Process PDF", type="primary"):
            if not gemini_api_key:
                st.error("Please provide your Gemini API key first!")
            else:
                with st.spinner("Processing PDF..."):
                    try:
                        # Extract text
                        pdf_text = extract_text_from_pdf(uploaded_file)
                        
                        if pdf_text.strip():
                            # Chunk text
                            st.session_state.text_chunks = chunk_text(pdf_text)
                            
                            # Load embeddings model
                            if st.session_state.embeddings_model is None:
                                with st.spinner("Loading embeddings model..."):
                                    st.session_state.embeddings_model = load_embeddings_model()
                            
                            # Create vector store
                            with st.spinner("Creating vector embeddings..."):
                                st.session_state.vector_store = create_vector_store(
                                    st.session_state.text_chunks, 
                                    st.session_state.embeddings_model
                                )
                            
                            st.session_state.pdf_processed = True
                            st.success(f"✅ PDF processed successfully! Created {len(st.session_state.text_chunks)} text chunks.")
                            
                            # Show preview of first chunk
                            with st.expander("📖 Preview of extracted text"):
                                st.write(st.session_state.text_chunks[0][:500] + "..." if len(st.session_state.text_chunks[0]) > 500 else st.session_state.text_chunks[0])
                                
                        else:
                            st.error("Could not extract text from the PDF. Please check if the PDF contains readable text.")
                    
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")

with col2:
    st.header("💬 Ask Questions")
    
    # Show processing status
    if st.session_state.pdf_processed:
        st.success("✅ PDF is ready for questions!")
        
        # Enable Q&A checkbox
        enable_qa = st.checkbox("🤖 Enable Question Answering", value=True)
        
        if enable_qa and gemini_api_key:
            # Question input
            user_question = st.text_area(
                "Ask a question about the PDF:",
                placeholder="What is the main topic of this document?",
                height=100
            )
            
            # Suggested questions
            st.markdown("**💡 Suggested questions:**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("📋 Summarize the document"):
                    user_question = "Please provide a comprehensive summary of this document."
                if st.button("🔍 Key findings"):
                    user_question = "What are the key findings or main points in this document?"
            
            with col_b:
                if st.button("📊 Important data"):
                    user_question = "What important data, statistics, or numbers are mentioned?"
                if st.button("🎯 Main conclusions"):
                    user_question = "What are the main conclusions or recommendations?"
            
            if st.button("🔍 Get Answer", type="primary") or user_question != st.session_state.get('last_question', ''):
                if user_question.strip():
                    st.session_state['last_question'] = user_question
                    
                    with st.spinner("Searching and generating answer..."):
                        try:
                            # Search for relevant chunks
                            relevant_chunks = search_similar_chunks(
                                user_question,
                                st.session_state.vector_store,
                                st.session_state.text_chunks,
                                st.session_state.embeddings_model,
                                k=5  # Increased to get more context
                            )
                            
                            if relevant_chunks:
                                # Generate answer
                                answer = generate_answer(
                                    user_question,
                                    relevant_chunks,
                                    gemini_api_key,
                                    model_option
                                )
                                
                                # Display results
                                st.subheader("🎯 Answer:")
                                st.write(answer)
                                
                                # Show confidence and source info
                                st.info(f"📊 Based on {len(relevant_chunks)} relevant text chunks from the document")
                                
                                # Show relevant context (optional)
                                with st.expander("📖 Source Context"):
                                    for i, chunk in enumerate(relevant_chunks):
                                        st.write(f"**Chunk {i+1}:**")
                                        st.write(chunk[:800] + "..." if len(chunk) > 800 else chunk)
                                        if i < len(relevant_chunks) - 1:
                                            st.divider()
                            else:
                                st.warning("❌ No relevant context found for your question. Try rephrasing or ask about different aspects of the document.")
                        
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                else:
                    st.warning("Please enter a question!")
        
        elif not enable_qa:
            st.info("☑️ Check the box above to enable question answering.")
        elif not gemini_api_key:
            st.warning("🔑 Please provide your Gemini API key to ask questions.")
    
    else:
        st.info("👆 Please upload and process a PDF first to enable question answering.")

# Footer
st.divider()
st.markdown("""
### 🚀 How it works:
1. **Upload PDF**: Choose a PDF document to analyze
2. **Process**: The app extracts text and creates vector embeddings using sentence transformers
3. **Enable Q&A**: Use the checkbox to enable question answering
4. **Ask Questions**: Ask questions about the content and get AI-powered answers
5. **View Sources**: Expand the source context to see which parts of the document were used

### 🔧 Technical Details:
- **Vector Embeddings**: Uses `all-MiniLM-L6-v2` model for creating semantic embeddings
- **Vector Search**: FAISS for efficient similarity search
- **AI Model**: Google Gemini for generating contextual answers
- **Session Storage**: All data is stored in session memory (not saved to backend)

### 🎯 Tips for Better Results:
- Ask specific questions rather than very broad ones
- Try different phrasings if you don't get the expected answer
- Use the suggested question buttons for common queries
""")

# Display session info in sidebar
with st.sidebar:
    if st.session_state.pdf_processed:
        st.divider()
        st.header("📊 Session Info")
        st.metric("Text Chunks", len(st.session_state.text_chunks))
        st.metric("Vector Store", "✅ Active" if st.session_state.vector_store else "❌ Inactive")
        st.metric("Embeddings Model", "✅ Loaded" if st.session_state.embeddings_model else "❌ Not Loaded")
        
        # Clear session button
        if st.button("🗑️ Clear Session", help="Clear all processed data and start fresh"):
            st.session_state.vector_store = None
            st.session_state.text_chunks = []
            st.session_state.embeddings_model = None
            st.session_state.pdf_processed = False
            if 'last_question' in st.session_state:
                del st.session_state['last_question']
            st.rerun()
    
    