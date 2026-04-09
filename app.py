import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# --- Page Config ---
st.set_page_config(page_title="RAG Lens", layout="wide")

# --- UI Header ---
st.title("RAG Insight: Chunking & Retrieval Visualizer")
st.subheader("Advanced Retrieval Observability & Hallucination Guard")

# --- 1. API & Secrets Management ---
# Try to get key from Streamlit Secrets (Private)
default_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("🔑 API Setup")
    # We leave 'value' empty so your key is never exposed in the UI
    user_provided_key = st.text_input(
        "Use your own Groq Key (Optional)", 
        type="password",
        help="Leave blank to use the app's default key. Enter yours if the existing key isn't working."
    )
    
    # Decision Logic: User key takes priority, then your secret key
    active_api_key = user_provided_key if user_provided_key else default_key

    if not active_api_key:
        st.error("No API Key found. Please enter one to start.")
        st.stop()
    
    st.header("⚙️ RAG Configuration")
    strategy = st.radio("Chunking Strategy", ["Fixed-Size", "Semantic (Advanced)"])
    
    if strategy == "Fixed-Size":
        size = st.slider("Chunk Size", 200, 1500, 500)
        overlap = st.slider("Overlap", 0, 200, 50)
    
    st.header("🛡️ Verification")
    enable_guard = st.toggle("Hallucination Guard", value=True)
    st.caption("Uses a second LLM pass to verify facts against context.")

# --- 2. Title & Onboarding ---
st.title("🔍 RAG Lens")
st.markdown("### Advanced Retrieval Observability & Debugging Suite")

# --- 3. Document Loading (Upload or Default) ---
uploaded_file = st.file_uploader("Upload a Technical PDF", type="pdf")

# Use a default file if nothing is uploaded
target_file = "sample_docs.pdf" 

if uploaded_file:
    target_file = "temp_upload.pdf"
    with open(target_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Custom document loaded!")
elif os.path.exists("sample_docs.pdf"):
    st.info("💡 Pro-Tip: I've pre-loaded a 'RAG Best Practices' doc for you to test.")
else:
    st.warning("Please upload a PDF or add 'sample_docs.pdf' to your repo to use the default mode.")
    st.stop()

# --- 4. Processing Core ---
if active_api_key:
    try:
        # Load Embeddings (Local CPU)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM (Groq Llama 3 70B for high logic)
        llm = ChatGroq(groq_api_key=active_api_key, model_name="openai/gpt-oss-20b")

        # Load and Split
        loader = PyPDFLoader(target_file)
        raw_docs = loader.load()

        if strategy == "Fixed-Size":
            splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        else:
            with st.spinner("Calculating semantic boundaries... (This takes a moment on CPU)"):
                splitter = SemanticChunker(embeddings)
        
        chunks = splitter.split_documents(raw_docs)

         # --- Debug: Chunk Visualization ---
        with st.expander("👀 View Raw Chunks (The Data Layer)"):
            chunk_df = pd.DataFrame([{"Content": c.page_content, "Length": len(c.page_content)} for c in chunks])
            st.table(chunk_df.head(10))
        
        # Create Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # --- 5. Guided Query UI ---
        st.divider()
        sample_q = "What are the trade-offs between different chunking sizes in RAG?"
        query = st.text_input("Ask a question about the document:", value=sample_q)

        if query:
            with st.spinner("🔍 Analyzing retrieval and verifying..."):
                # A. Retrieval
                retrieved_docs = vectorstore.similarity_search(query, k=3)
                context_text = "\n\n".join([d.page_content for d in retrieved_docs])

                # B. Generation with Rate Limit Catch
                try:
                    main_prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer using ONLY the context:"
                    initial_answer = llm.invoke(main_prompt).content

                    # C. Hallucination Guard
                    final_answer = initial_answer
                    if enable_guard:
                        guard_prompt = f"Verify this answer against the context. If it adds info not in context, fix it.\nContext: {context_text}\nAnswer: {initial_answer}"
                        final_answer = llm.invoke(guard_prompt).content

                    # --- 6. Visualizer Display ---
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("🤖 AI Response")
                        st.write(final_answer)
                        if enable_guard:
                            st.caption("🛡️ Verified by Hallucination Guard Layer")

                    with col2:
                        st.subheader("📋 Debug: Retrieval Context")
                        for i, doc in enumerate(retrieved_docs):
                            with st.expander(f"Chunk {i+1} | Source: Page {doc.metadata.get('page', '??')}"):
                                st.write(doc.page_content)

                except Exception as e:
                    if "rate_limit" in str(e).lower():
                        st.error("🛑 Groq Rate Limit Reached! Use your own API key in the sidebar to continue.")
                    else:
                        st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Enter your Groq API Key in the sidebar to activate the LLM.")