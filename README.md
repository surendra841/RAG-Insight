# 🔍 RAG Lens | Advanced Retrieval Observability

**RAG Lens** is a high-fidelity observability tool designed to strip away the "black box" of Retrieval-Augmented Generation. While most RAG applications only show the final output, RAG Lens allows engineers to visualize the entire data lifecycle—from semantic chunking to verified generation.

## 🚀 Key Features

* **Dual-Strategy Chunking:** Compare standard **Fixed-Size** character splitting against **Semantic Chunking** (AI-driven sentence similarity) to see how data structure affects retrieval.
* **Retrieval Transparency:** Side-by-side UI that displays the exact document chunks retrieved from the vector store before they are passed to the LLM.
* **Hallucination Guard:** Implements a multi-agent "Fact-Checker" pattern that uses a second LLM pass to verify the final response against the source context.
* **Enterprise-Ready Security:** Secure credential management with an automated fallback system (Streamlit Secrets vs. User-provided keys).
* **Instant Onboarding:** Includes a pre-loaded technical dataset on RAG Best Practices for immediate testing and demonstration.

## 🛠️ The Tech Stack

* **Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM Inference:** [Groq](https://groq.com/) (Llama 3 70B)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Vector Database:** FAISS (Local CPU)
* **Frontend:** Streamlit

## ⚙️ Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/rag-lens.git](https://github.com/your-username/rag-lens.git)
   cd rag-lens
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Configure Secrets**
   Create a .streamlit/secrets.toml file and inside the file add your Groq API key:
   ```bash
   GROQ_API_KEY = "your_grok_api_key_here"
4. **Run the app**
   ```bash
   streamlit run app.py
## 🧠 Why I Built This

As a senior engineer, I realized that the biggest challenge in production AI isn't just getting an answer—it's getting a **reliable** answer. I built RAG Lens to provide a debugging suite that helps developers:

1. **Visualize** how different embedding models "see" their data.
2. **Debug** "Noise in Context" by seeing exactly what chunks are retrieved.
3. **Minimize** hallucinations using an automated verification layer.

---

## 🛡️ Security Note

This project uses `.gitignore` to ensure that local `.streamlit/secrets.toml` files are never committed to version control, following DevSecOps best practices.
