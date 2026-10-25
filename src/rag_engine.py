"""
Module: src/rag_engine.py
Description: Retrieval-Augmented Generation (RAG) engine featuring:
             - Semantic top-k similarity retrieval with score normalization
             - Strict anti-hallucination prompt engineering
             - Multi-provider support: Google Gemini API, Groq Llama-3 API,
               and 100% Offline Local Extractive / Flan-T5 mode.
"""

import os
import re
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Silence benign upstream SDK migration warnings in terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

# LLM SDK Imports with resilient fallbacks
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with provenance metadata and similarity score."""
    content: str
    source: str
    page: int
    score: float  # Normalized similarity score (0.0 to 1.0)
    chunk_id: str
    rank: int

    @property
    def score_percentage(self) -> str:
        """Returns human-friendly percentage string (e.g. '94.2%')."""
        return f"{self.score * 100:.1f}%"

    @property
    def citation(self) -> str:
        """Returns standard citation tag."""
        return f"[Source: {self.source}, Page: {self.page}]"


@dataclass
class RAGResponse:
    """Structured response container returned by the RAG engine."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    provider: str
    model_name: str
    latency_sec: float
    context_text: str = ""


# ==============================================================================
# Strict Anti-Hallucination System Prompt
# ==============================================================================

RAG_SYSTEM_PROMPT = """You are a Senior Technical Document Q&A Assistant operating under strict factual grounding rules.

CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATIONS:
1. Grounding Rule: Answer the question ONLY using the factual context chunks provided below. DO NOT assume, extrapolate, or use pre-trained external knowledge.
2. Refusal Protocol: If the provided context chunks DO NOT contain sufficient evidence to answer the query directly, you MUST answer:
   "Based on the provided document(s), this information is not available."
   You may mention what related topics were found, but NEVER fabricate answers.
3. Citation Requirement: For EVERY statement, claim, or data point you make, you MUST cite its exact source immediately inline in this format:
   [Source: <filename>, Page: <page_number>]
4. Formatting: Write in clear, professional Markdown with bullet points or numbered lists where appropriate for technical readability.
"""

USER_PROMPT_TEMPLATE = """CONTEXT CHUNKS:
----------------------------------------
{context_blocks}
----------------------------------------

USER QUESTION: {query}

Please provide a precise, grounded answer with inline citations [Source: filename, Page: X] for each claim:"""


# ==============================================================================
# Retrieval & Context Formatting
# ==============================================================================

def normalize_faiss_score(raw_score: float, is_distance: bool = True) -> float:
    """
    Normalizes FAISS scores to a 0.0 - 1.0 similarity range.
    For L2 distance with normalized vectors:
        cosine_similarity = 1 - (L2_distance^2 / 2)
    For cosine/inner product similarity: raw_score is already cosine similarity.
    """
    if is_distance:
        # Distance: lower is more similar.
        # When vectors are unit-normalized, L2 distance ranges between 0.0 (identical) and 2.0 (opposite).
        cos_sim = 1.0 - ((raw_score ** 2) / 2.0)
        return max(0.0, min(1.0, float(cos_sim)))
    else:
        # Inner product / Cosine similarity: higher is more similar.
        return max(0.0, min(1.0, float(raw_score)))


def retrieve_relevant_chunks(
    vectorstore: Any,
    query: str,
    top_k: int = 4
) -> List[RetrievedChunk]:
    """
    Executes semantic similarity search over the FAISS vectorstore and maps results
    to structured RetrievedChunk instances with normalized scores.

    Args:
        vectorstore: FAISS vectorstore instance.
        query: User question string.
        top_k: Number of chunks to retrieve.

    Returns:
        List of RetrievedChunk instances sorted by relevance.
    """
    if not vectorstore:
        raise ValueError("Vectorstore is not initialized. Please index documents first.")

    # Execute similarity search with distance score
    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    retrieved: List[RetrievedChunk] = []
    for rank, (doc, score) in enumerate(results_with_scores, 1):
        # LangChain FAISS returns Euclidean (L2) distance by default
        norm_score = normalize_faiss_score(score, is_distance=True)

        meta = doc.metadata or {}
        chunk = RetrievedChunk(
            content=doc.page_content.strip(),
            source=meta.get("source", "Unknown Document"),
            page=meta.get("page", 1),
            score=norm_score,
            chunk_id=meta.get("chunk_id", f"chunk_{rank}"),
            rank=rank
        )
        retrieved.append(chunk)

    return retrieved


def format_context_blocks(chunks: List[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks into clear, demarcated context blocks for the LLM prompt.
    """
    formatted_blocks = []
    for chunk in chunks:
        block = (
            f"--- CHUNK {chunk.rank} [Source: {chunk.source}, Page: {chunk.page}, Match: {chunk.score_percentage}] ---\n"
            f"{chunk.content}"
        )
        formatted_blocks.append(block)
    return "\n\n".join(formatted_blocks)


# ==============================================================================
# LLM Generation Providers
# ==============================================================================

def generate_gemini_response(
    query: str,
    context_blocks: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """
    Generates response using Google Gemini API with system instructions.
    """
    if genai is None:
        raise ImportError("google-generativeai is required. Run: pip install google-generativeai")

    genai.configure(api_key=api_key)

    # Use generation config for deterministic, grounded reasoning
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_output_tokens": 1500,
    }

    prompt = USER_PROMPT_TEMPLATE.format(context_blocks=context_blocks, query=query)

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=RAG_SYSTEM_PROMPT,
            generation_config=generation_config
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Fallback to gemini-1.5-flash if gemini-2.5-flash is not available in region
        if "2.5" in model_name:
            try:
                fallback_model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=RAG_SYSTEM_PROMPT,
                    generation_config=generation_config
                )
                response = fallback_model.generate_content(prompt)
                return response.text.strip()
            except Exception:
                pass
        raise RuntimeError(f"Gemini API Error: {str(e)}") from e


def generate_groq_response(
    query: str,
    context_blocks: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generates response using Groq Cloud API for ultra-fast Llama-3 inference.
    """
    if Groq is None:
        raise ImportError("groq is required. Run: pip install groq")

    client = Groq(api_key=api_key)
    prompt = USER_PROMPT_TEMPLATE.format(context_blocks=context_blocks, query=query)

    # Common Groq models fallback list
    candidates = [model_name, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    last_error = None
    for candidate in candidates:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=candidate,
                temperature=0.1,
                max_tokens=1500
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Groq API Error: {str(last_error)}")


def generate_local_extractive_response(
    query: str,
    chunks: List[RetrievedChunk]
) -> str:
    """
    100% Offline Local Extractive Generation & Evidence Synthesis.
    Operates without any API key or external service. Analyzes retrieved chunks,
    extracts the most salient sentences answering the query, and formats an answer
    with exact inline page citations.
    """
    if not chunks:
        return "Based on the provided document(s), this information is not available."

    query_tokens = set(re.findall(r'\w+', query.lower()))
    # Remove trivial stopwords
    stopwords = {"what", "is", "the", "in", "and", "or", "of", "to", "for", "a", "an", "how", "why", "does", "are"}
    key_query_tokens = query_tokens - stopwords

    scored_sentences = []

    for chunk in chunks:
        # Split chunk into sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 20:
                continue
            s_tokens = set(re.findall(r'\w+', s_clean.lower()))
            overlap = len(key_query_tokens.intersection(s_tokens))
            if overlap > 0:
                # Rank sentence by query overlap and parent chunk similarity score
                combined_score = (overlap * 2.0) + (chunk.score * 1.5)
                scored_sentences.append({
                    "sentence": s_clean,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": combined_score,
                    "citation": chunk.citation
                })

    # Sort sentences by relevance
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)

    if not scored_sentences:
        # If no direct keyword overlap found, return executive summary of top chunk
        top_chunk = chunks[0]
        return (
            f"**Offline Extractive Synthesis** (No API Key Required)\n\n"
            f"Direct keyword correlation was weak for this query, but the highest-ranked semantic passage retrieved is:\n\n"
            f"> \"{top_chunk.content[:350]}...\"\n\n"
            f"**Source Reference:** {top_chunk.citation} (Semantic Match: {top_chunk.score_percentage})."
        )

    # Take top 3 most relevant sentences avoiding duplicate ideas
    selected = []
    seen = set()
    for s_item in scored_sentences:
        s_text = s_item["sentence"]
        if s_text not in seen:
            seen.add(s_text)
            selected.append(s_item)
        if len(selected) >= 3:
            break

    response_lines = [
        "**Offline Local Extractive Response** *(Synthesized without external API keys)*:\n"
    ]
    for item in selected:
        response_lines.append(f"- {item['sentence']} **{item['citation']}**")

    response_lines.append(
        "\n> *Tip: To enable abstractive reasoning and conversational synthesis, select Google Gemini or Groq in the sidebar.*"
    )

    return "\n".join(response_lines)


# ==============================================================================
# Unified RAG Engine Pipeline
# ==============================================================================

class RAGEngine:
    """
    Production RAG Coordinator managing semantic retrieval, anti-hallucination prompt assembly,
    and dispatching to free LLM providers (Gemini, Groq) or local offline extractive mode.
    """

    SUPPORTED_PROVIDERS = [
        "Google Gemini (Free API)",
        "Groq Llama-3 (Free API)",
        "Offline Local Mode (No API Key Required)"
    ]

    DEFAULT_MODELS = {
        "Google Gemini (Free API)": "gemini-2.5-flash",
        "Groq Llama-3 (Free API)": "llama-3.3-70b-versatile",
        "Offline Local Mode (No API Key Required)": "Extractive Synthesis / Local"
    }

    def __init__(self, vectorstore: Any):
        self.vectorstore = vectorstore

    def query(
        self,
        user_query: str,
        provider: str = "Offline Local Mode (No API Key Required)",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: int = 4
    ) -> RAGResponse:
        """
        Executes complete RAG workflow:
        1. Semantic retrieval with FAISS score normalization.
        2. Prompt construction with strict anti-hallucination guardrails.
        3. LLM generation with inline citations.
        """
        start_time = time.time()

        # Step 1: Retrieve context chunks
        chunks = retrieve_relevant_chunks(self.vectorstore, user_query, top_k=top_k)
        if not chunks:
            return RAGResponse(
                query=user_query,
                answer="Based on the provided document(s), no relevant information was found.",
                retrieved_chunks=[],
                provider=provider,
                model_name=model_name or "N/A",
                latency_sec=round(time.time() - start_time, 2),
                context_text=""
            )

        context_blocks = format_context_blocks(chunks)

        # Step 2: Route to selected Provider
        selected_model = model_name or self.DEFAULT_MODELS.get(provider, "default")
        answer = ""

        if provider == "Google Gemini (Free API)":
            active_key = api_key or os.getenv("GEMINI_API_KEY")
            if not active_key:
                fallback_ans = generate_local_extractive_response(user_query, chunks)
                answer = (
                    "⚠️ **Gemini API Key Missing**: Please paste your free Gemini API key in the sidebar.\n\n"
                    f"{fallback_ans}"
                )
            else:
                answer = generate_gemini_response(
                    query=user_query,
                    context_blocks=context_blocks,
                    api_key=active_key,
                    model_name=selected_model
                )

        elif provider == "Groq Llama-3 (Free API)":
            active_key = api_key or os.getenv("GROQ_API_KEY")
            if not active_key:
                fallback_ans = generate_local_extractive_response(user_query, chunks)
                answer = (
                    "⚠️ **Groq API Key Missing**: Please paste your free Groq API key in the sidebar.\n\n"
                    f"{fallback_ans}"
                )
            else:
                answer = generate_groq_response(
                    query=user_query,
                    context_blocks=context_blocks,
                    api_key=active_key,
                    model_name=selected_model
                )

        else:
            # Offline Local Mode
            answer = generate_local_extractive_response(user_query, chunks)

        elapsed = round(time.time() - start_time, 2)

        return RAGResponse(
            query=user_query,
            answer=answer,
            retrieved_chunks=chunks,
            provider=provider,
            model_name=selected_model,
            latency_sec=elapsed,
            context_text=context_blocks
        )

