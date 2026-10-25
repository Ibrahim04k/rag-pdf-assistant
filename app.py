"""
Smart Document Q&A Assistant (RAG & Generative AI)
Interactive Conversational Streamlit UI with PDF Ingestion, Multi-Provider LLMs,
Local Offline Fallback, and Provenance Chunk Inspector.
"""

import os
import sys

# Auto-relaunch inside Streamlit if executed directly via `python app.py` (e.g. VS Code ▶ Run button)
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as _st_exists
        _in_streamlit = _st_exists()
    except Exception:
        _in_streamlit = False

    if not _in_streamlit:
        import subprocess
        print("Launching Streamlit application in browser...")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]))

import io
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Internal modular imports
from src.ingestion import (
    process_pdf_documents,
    extract_text_from_pdf,
    split_documents,
    get_embedding_model,
    build_faiss_index,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.rag_engine import (
    RAGEngine,
    RetrievedChunk,
    RAGResponse,
)
from sample_docs.generate_sample_doc import ensure_sample_pdf_exists, SAMPLE_PDF_PATH

# ==============================================================================
# Streamlit Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="Smart Document Q&A Assistant | RAG & Generative AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise look & feel
st.markdown("""
<style>
    /* Global Typography & Spacing */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        color: #f8fafc;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Metric Badges */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Source Chunk Inspector Card */
    .chunk-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-weight: 600;
        color: #1e293b;
    }
    .badge-score {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Meta badge */
    .meta-tag {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.75rem;
        color: #475569;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

if "active_file_names" not in st.session_state:
    st.session_state.active_file_names = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ==============================================================================
# Sidebar: Document Management & LLM Configuration
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=56)
    st.title("Control Center")
    st.caption("Production RAG & Vector Search Configuration")

    st.divider()

    # ---------------------------------------------------------
    # 1. LLM Provider Selector
    # ---------------------------------------------------------
    st.subheader("1. LLM Provider")

    provider_options = [
        "Offline Local Mode (No API Key Required)",
        "Google Gemini (Free API)",
        "Groq Llama-3 (Free API)"
    ]

    selected_provider = st.selectbox(
        "Inference Engine",
        options=provider_options,
        index=0,
        help="Select between 100% offline local inference (no API key required) or free cloud API tiers."
    )

    # Dynamic Model Selector
    if selected_provider == "Google Gemini (Free API)":
        model_options = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        selected_model = st.selectbox("Gemini Model", options=model_options, index=0)
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get free API key at: https://aistudio.google.com/app/apikey"
        )
        if not api_key_input:
            st.info("💡 Paste your free Gemini key or switch to **Offline Local Mode** below.")

    elif selected_provider == "Groq Llama-3 (Free API)":
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        selected_model = st.selectbox("Groq Model", options=model_options, index=0)
        env_key = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=env_key,
            type="password",
            placeholder="gsk_...",
            help="Get free API key at: https://console.groq.com/keys"
        )
        if not api_key_input:
            st.info("💡 Paste your free Groq key or switch to **Offline Local Mode** below.")

    else:
        # Offline Local Mode
        selected_model = "Local Extractive Synthesizer"
        api_key_input = ""
        st.success("✅ **100% Offline Mode Active**: No API key or cloud network access required!")

    st.divider()

    # ---------------------------------------------------------
    # 2. Document Upload & Ingestion
    # ---------------------------------------------------------
    st.subheader("2. Document Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Technical PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple PDF documents to index into the local FAISS vectorstore."
    )

    # 1-Click Sample Document Button
    col_sample, col_clear = st.columns([1.8, 1.2])
    with col_sample:
        load_sample = st.button("⚡ Load Sample AI Paper", use_container_width=True, help="Load 4-page Generative AI technical whitepaper instantly")
    with col_clear:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.rag_engine = None
            st.session_state.doc_stats = None
            st.session_state.active_file_names = []
            st.rerun()

    # ---------------------------------------------------------
    # 3. Advanced Retrieval Tuning (Expander)
    # ---------------------------------------------------------
    with st.expander("⚙️ Advanced Retrieval Hyperparameters", expanded=False):
        top_k_slider = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=4, step=1)
        chunk_size_slider = st.slider("Chunk Size (characters)", min_value=300, max_value=1500, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap_slider = st.slider("Chunk Overlap (characters)", min_value=50, max_value=300, value=DEFAULT_CHUNK_OVERLAP, step=25)
        st.caption(f"Dense Embeddings: `{DEFAULT_EMBEDDING_MODEL}` (384-d, L2/Cosine normalized).")

    # Ingestion Trigger
    process_btn = st.button("🚀 Process & Index Documents", type="primary", use_container_width=True)
    if process_btn and not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file above, or click '⚡ Load Sample AI Paper' to test instantly!")

    if load_sample or (process_btn and uploaded_files):
        with st.status("Building Vector Index...", expanded=True) as status:
            try:
                target_sources = []
                target_names = []

                if load_sample:
                    status.write("📄 Verifying sample technical paper...")
                    sample_path = ensure_sample_pdf_exists(SAMPLE_PDF_PATH)
                    target_sources.append(sample_path)
                    target_names.append("sample_ai_paper.pdf")
                else:
                    for f in uploaded_files:
                        target_sources.append(f)
                        target_names.append(f.name)

                status.write(f"🔍 Parsing PDF text page-by-page ({len(target_sources)} document(s))...")
                all_pages = []
                for src in target_sources:
                    pages = extract_text_from_pdf(src)
                    all_pages.extend(pages)

                status.write(f"🧩 Recursive chunking (Size: {chunk_size_slider}, Overlap: {chunk_overlap_slider})...")
                chunks = split_documents(all_pages, chunk_size=chunk_size_slider, chunk_overlap=chunk_overlap_slider)

                status.write(f"🧠 Generating local dense embeddings (`{DEFAULT_EMBEDDING_MODEL}`)...")
                emb_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)

                status.write("⚡ Indexing vectors into FAISS...")
                vectorstore = build_faiss_index(chunks, emb_model)

                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_engine = RAGEngine(vectorstore)
                st.session_state.active_file_names = target_names
                st.session_state.doc_stats = {
                    "total_docs": len(target_sources),
                    "total_pages": len(all_pages),
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size_slider,
                    "chunk_overlap": chunk_overlap_slider
                }

                status.update(label="✅ Vector Index Successfully Built!", state="complete", expanded=False)
                st.toast(f"Indexed {len(chunks)} chunks across {len(all_pages)} pages!", icon="🚀")
                st.rerun()

            except Exception as e:
                status.update(label=f"❌ Ingestion Failed: {str(e)}", state="error", expanded=True)
                st.error(f"Error during document processing: {str(e)}")

    # ---------------------------------------------------------
    # 4. Active Index Metrics
    # ---------------------------------------------------------
    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Active Index Metrics")
        stats = st.session_state.doc_stats

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_pages']}</div>
                <div class="metric-label">Pages Indexed</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Vector Chunks</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top: 10px; font-size: 0.8rem; color: #64748b;">
            <b>Active Files:</b> {', '.join(st.session_state.active_file_names)}<br>
            <b>Index Backend:</b> FAISS IndexFlatL2<br>
            <b>Embeddings:</b> all-MiniLM-L6-v2 (Local)
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Main Application Interface
# ==============================================================================

def auto_index_sample_doc():
    """Helper to auto-index sample document for immediate testing."""
    sample_path = ensure_sample_pdf_exists(SAMPLE_PDF_PATH)
    pages = extract_text_from_pdf(sample_path)
    chunks = split_documents(pages, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    emb_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vectorstore = build_faiss_index(chunks, emb_model)
    st.session_state.vectorstore = vectorstore
    st.session_state.rag_engine = RAGEngine(vectorstore)
    st.session_state.active_file_names = ["sample_ai_paper.pdf"]
    st.session_state.doc_stats = {
        "total_docs": 1,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP
    }


# Top Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>📑 Smart Document Q&A Assistant</span>
    </div>
    <div class="hero-subtitle">
        Enterprise Retrieval-Augmented Generation (RAG) system with local Hugging Face vector embeddings,
        strict anti-hallucination protocols, transparent source chunk inspection, and multi-provider LLM support.
    </div>
</div>
""", unsafe_allow_html=True)

# Check if index is ready
if not st.session_state.vectorstore:
    st.info("👈 **Get Started:** Upload your PDF files in the sidebar, or click **'⚡ Load Sample AI Paper'** for instant 1-click testing!")

    st.markdown("### 💡 What makes this RAG architecture production-grade?")
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown("""
        **🔒 Anti-Hallucination Guardrails**
        - Refusal protocol for ungrounded queries
        - Strict inline provenance citations: `[Source: X, Page: Y]`
        - Deterministic low-temperature sampling
        """)
    with col_feat2:
        st.markdown("""
        **⚡ Local Embeddings & Vectorstore**
        - `all-MiniLM-L6-v2` dense embeddings
        - Fast in-memory FAISS similarity search
        - 100% free to run without vector DB subscriptions
        """)
    with col_feat3:
        st.markdown("""
        **🌐 Flexible Multi-Provider LLMs**
        - Google Gemini (Free API tier)
        - Groq Llama-3.3-70B (Sub-second inference)
        - 100% Offline Local Extractive Fallback
        """)

    # Quick test sample questions if sample document is mentioned
    st.markdown("---")
    st.markdown("#### 🎯 Instant Sample Questions (Click to auto-index and test with Sample AI Paper):")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("📌 What chunk overlap is recommended for dense vector retrieval?"):
            auto_index_sample_doc()
            st.session_state.pending_query = "What chunk overlap is recommended for dense vector retrieval?"
            st.rerun()
        if st.button("📌 Compare FAISS IndexFlatL2 versus HNSW for vector search."):
            auto_index_sample_doc()
            st.session_state.pending_query = "Compare FAISS IndexFlatL2 versus HNSW for vector search."
            st.rerun()
    with q_col2:
        if st.button("📌 What are the four core metrics used in the RAGAS evaluation framework?"):
            auto_index_sample_doc()
            st.session_state.pending_query = "What are the four core metrics used in the RAGAS evaluation framework?"
            st.rerun()
        if st.button("📌 What is the latency advantage of Groq LPU architecture for Llama-3.3-70B?"):
            auto_index_sample_doc()
            st.session_state.pending_query = "What is the latency advantage of Groq LPU architecture for Llama-3.3-70B?"
            st.rerun()

else:
    # If a pending query was selected from quick buttons, auto-run it
    auto_query = None
    if st.session_state.pending_query:
        auto_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # If message contains retrieved chunks, display expandable provenance inspector
            if message.get("chunks"):
                chunks: List[RetrievedChunk] = message["chunks"]
                with st.expander(f"📚 View Retrieved Source Chunks & Page Citations ({len(chunks)} chunks used)", expanded=False):
                    for chunk in chunks:
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span>📄 <b>{chunk.source}</b> (Page {chunk.page})</span>
                                <span class="badge-score">Rank #{chunk.rank} &bull; Match: {chunk.score_percentage}</span>
                            </div>
                            <div style="font-style: italic; color: #334155; line-height: 1.45;">
                                "{chunk.content}"
                            </div>
                            <div class="meta-tag">Chunk ID: {chunk.chunk_id}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if message.get("meta"):
                meta = message["meta"]
                st.caption(f"⚡ Latency: {meta.get('latency', 'N/A')}s &bull; Provider: {meta.get('provider')} &bull; Model: {meta.get('model')}")

    # Chat Input Box
    user_query = st.chat_input("Ask a technical question about your indexed document(s)...") or auto_query

    if user_query:
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching vector index & generating grounded answer..."):
                engine = st.session_state.rag_engine or RAGEngine(st.session_state.vectorstore)
                
                # Retrieve current top_k setting from slider
                active_top_k = top_k_slider if "top_k_slider" in locals() else 4

                try:
                    response: RAGResponse = engine.query(
                        user_query=user_query,
                        provider=selected_provider,
                        api_key=api_key_input,
                        model_name=selected_model,
                        top_k=active_top_k
                    )

                    st.markdown(response.answer)

                    # Display Retrieved Chunks Expander
                    if response.retrieved_chunks:
                        with st.expander(f"📚 View Retrieved Source Chunks & Page Citations ({len(response.retrieved_chunks)} chunks used)", expanded=True):
                            for chunk in response.retrieved_chunks:
                                st.markdown(f"""
                                <div class="chunk-card">
                                    <div class="chunk-header">
                                        <span>📄 <b>{chunk.source}</b> (Page {chunk.page})</span>
                                        <span class="badge-score">Rank #{chunk.rank} &bull; Match: {chunk.score_percentage}</span>
                                    </div>
                                    <div style="font-style: italic; color: #334155; line-height: 1.45;">
                                        "{chunk.content}"
                                    </div>
                                    <div class="meta-tag">Chunk ID: {chunk.chunk_id}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.caption(f"⚡ Latency: {response.latency_sec}s &bull; Provider: {response.provider} &bull; Model: {response.model_name}")

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "chunks": response.retrieved_chunks,
                        "meta": {
                            "latency": response.latency_sec,
                            "provider": response.provider,
                            "model": response.model_name
                        }
                    })

                except Exception as err:
                    st.error(f"Generation Error: {str(err)}")
                    st.info("💡 You can switch to **Offline Local Mode** in the sidebar to bypass cloud API errors.")
