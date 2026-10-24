"""
Sample Document Generator for RAG Demonstration
Generates a realistic, 4-page technical whitepaper on Generative AI & RAG
to enable instant 1-click testing of the Q&A Assistant without requiring user file uploads.
"""

import os
import sys

SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "sample_ai_paper.pdf")

PAGES_CONTENT = [
    {
        "title": "Modern Retrieval-Augmented Generation (RAG) Architecture",
        "page_num": 1,
        "sections": [
            (
                "1. Executive Summary & Overview",
                "Retrieval-Augmented Generation (RAG) has emerged as the industry standard architectural pattern "
                "for grounding Large Language Models (LLMs) in external, proprietary enterprise knowledge bases. "
                "While LLMs exhibit remarkable reasoning capabilities, they suffer from two critical limitations: "
                "parametric knowledge obsolescence and factual hallucinations. In RAG systems, rather than relying solely "
                "on static model weights, dynamic external documents are retrieved at inference time and supplied "
                "within the LLM context window as verified evidence."
            ),
            (
                "2. The Document Ingestion & Chunking Pipeline",
                "The ingestion pipeline transforms raw unstructured documents (PDFs, Markdown, Docx) into clean, searchable vectors. "
                "The standard recommended chunk size for sentence-transformer architectures is between 500 and 1000 characters, "
                "paired with a 15% to 20% chunk overlap (typically 100 to 150 characters). Recursive character text splitting "
                "is preferred because it respects natural paragraph boundaries, sentence stops, and whitespace hierarchy, "
                "thereby preserving local context across contiguous chunk boundaries."
            ),
            (
                "3. Dense Vector Embeddings",
                "Dense embeddings map semantic textual meaning into continuous high-dimensional vector spaces (e.g., 384 dimensions "
                "for all-MiniLM-L6-v2, and 1536 dimensions for OpenAI text-embedding-3-small). "
                "Unlike sparse lexical methods like BM25 which rely on exact keyword matches, dense vectors capture semantic synonyms, "
                "cross-lingual equivalence, and conceptual relationships through cosine similarity."
            )
        ]
    },
    {
        "title": "Vector Databases, Indexing, and Retrieval Dynamics",
        "page_num": 2,
        "sections": [
            (
                "4. Vector Indexing: FAISS IndexFlatL2 vs. HNSW",
                "Facebook AI Similarity Search (FAISS) provides highly optimized vector operations for GPU and CPU architectures. "
                "The IndexFlatL2 index performs exhaustive exact Euclidean distance search with zero recall loss, making it optimal "
                "for datasets with fewer than 50,000 vectors. For web-scale vector search exceeding millions of embeddings, "
                "Hierarchical Navigable Small World (HNSW) graphs offer logarithmic search complexity O(log N) by traversing "
                "multi-layer proximity graphs with approximate nearest neighbor (ANN) guarantees."
            ),
            (
                "5. Retrieval Top-K and Similarity Thresholds",
                "Standard RAG configurations query the vector database for the top-k most similar passages (typically k=3 to k=6). "
                "When cosine similarity falls below a critical threshold (e.g., 0.45 or distance > 1.10), the chunks are discarded "
                "to prevent context pollution. Context pollution occurs when irrelevant or noisy text confuses the attention mechanism "
                "of the generator LLM, leading to degraded synthesis quality."
            ),
            (
                "6. Two-Stage Retrieval with Cross-Encoder Re-Rankers",
                "Production enterprise systems frequently employ a two-stage retrieval pipeline: bi-encoders (such as all-MiniLM-L6-v2) "
                "rapidly retrieve the top-50 candidate documents in sub-10ms latency, followed by a computationally heavier "
                "cross-encoder (e.g., ms-marco-MiniLM-L-6-v2 or Cohere Rerank) that scores query-document token interactions "
                "jointly to select the final top-4 context chunks."
            )
        ]
    },
    {
        "title": "Hallucination Mitigation & Prompt Engineering Protocols",
        "page_num": 3,
        "sections": [
            (
                "7. Taxonomy of LLM Hallucinations",
                "Hallucinations in generative AI are broadly classified into two categories: "
                "Intrinsic hallucinations (where the generated output directly contradicts verified source context) and "
                "Extrinsic hallucinations (where the output introduces external claims that cannot be corroborated nor refuted "
                "from the source text). In mission-critical legal, medical, and financial domains, extrinsic hallucinations "
                "carry severe compliance risks."
            ),
            (
                "8. Prompt Engineering Guardrails for Factual Fidelity",
                "To eliminate ungrounded speculation, production system prompts enforce four fundamental constraints: "
                "1) Absolute domain boundary: The model is explicitly forbidden from utilizing prior parametric memory. "
                "2) Negative rejection protocol: When source evidence is insufficient, the model must output an exact fallback message. "
                "3) Atomic citation requirement: Every declarative statement must conclude with an immutable citation bracket [Source: X, Page: Y]. "
                "4) Low temperature setting: Setting temperature to 0.0 or 0.1 minimizes token sampling stochasticity."
            ),
            (
                "9. Evaluation Frameworks: RAGAS & TruLens",
                "Modern RAG quality assurance evaluates pipelines across four core metrics: Faithfulness (is the answer grounded in context?), "
                "Answer Relevance (does the answer address the user query?), Context Precision (are retrieved chunks signal-rich?), and "
                "Context Recall (did retrieval capture all necessary background knowledge?). TruLens and RAGAS automate these evaluations."
            )
        ]
    },
    {
        "title": "Enterprise Deployment, Security, and Scalability",
        "page_num": 4,
        "sections": [
            (
                "10. Latency Optimization and Streaming Inference",
                "User perceived latency is dominated by Time to First Token (TTFT). Utilizing modern inference accelerators such as "
                "Groq LPU (Language Processing Unit) architecture enables generation speeds surpassing 500 tokens per second for "
                "Llama-3.3-70B models. Streaming response generation allows client interfaces to render partial answers token-by-token, "
                "lowering perceived wait times below 300 milliseconds."
            ),
            (
                "11. Document Security & Access Control (RBAC)",
                "When indexing multi-tenant organizational documentation, vector embeddings must retain metadata tags detailing "
                "role-based access control (RBAC) permissions. Query filtering is executed at the vector database retrieval stage "
                "via metadata pre-filtering, preventing unauthorized documents from ever entering the prompt synthesis context."
            ),
            (
                "12. Conclusion and Future Directions",
                "The evolution of RAG systems is transitioning toward Agentic RAG, where autonomous agents dynamically decide "
                "when to retrieve, execute multi-hop reasoning over heterogeneous graph-vector hybrid stores, and self-reflect "
                "on output validity before presenting final conclusions to end users."
            )
        ]
    }
]


def generate_pdf_pure_python(output_path: str):
    """
    Generates a compliant, clean multi-page PDF document using standard PDF 1.4 specification
    with zero external binary or library dependencies.
    """
    objects = []
    
    def add_object(content: str) -> int:
        objects.append(content)
        return len(objects)

    # 1. Catalog (will be obj 1)
    # 2. Pages root (will be obj 2)
    # 3. Font (will be obj 3)
    # 4. Bold Font (will be obj 4)
    # Subsequent objects will be Page and Stream objects.

    total_pages = len(PAGES_CONTENT)
    page_obj_ids = []
    # Pre-calculate object ids:
    # Obj 1: Catalog
    # Obj 2: Pages root
    # Obj 3: Regular Font (Helvetica)
    # Obj 4: Bold Font (Helvetica-Bold)
    # For each page:
    #   Page Object: 5 + 2*i
    #   Contents Stream: 6 + 2*i
    for i in range(total_pages):
        page_obj_ids.append(5 + 2 * i)

    # Catalog
    catalog_str = "<< /Type /Catalog /Pages 2 0 R >>"
    # Pages Root
    kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_root_str = f"<< /Type /Pages /Kids [{kids_str}] /Count {total_pages} >>"
    # Fonts
    font_regular_str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    font_bold_str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"

    objects.append(catalog_str)       # 1
    objects.append(pages_root_str)    # 2
    objects.append(font_regular_str)  # 3
    objects.append(font_bold_str)     # 4

    # Build Pages and Contents
    for i, page_data in enumerate(PAGES_CONTENT):
        stream_id = 6 + 2 * i
        
        # Build text stream
        stream_ops = []
        # Header banner
        stream_ops.append("q")
        stream_ops.append("0.1 0.25 0.5 rg") # Deep blue
        stream_ops.append("50 740 512 30 re f")
        stream_ops.append("Q")

        # Header Title
        stream_ops.append("BT")
        stream_ops.append("/F2 14 Tf")
        stream_ops.append("1 1 1 rg") # White text
        stream_ops.append("60 750 Td")
        escaped_title = page_data["title"].replace("(", "\\(").replace(")", "\\)")
        stream_ops.append(f"({escaped_title}) Tj")
        stream_ops.append("ET")

        # Subtitle / Header metadata
        stream_ops.append("BT")
        stream_ops.append("/F1 9 Tf")
        stream_ops.append("0.4 0.4 0.4 rg") # Grey
        stream_ops.append("50 725 Td")
        stream_ops.append("(Technical Whitepaper | Research & Engineering Division | Page " + str(page_data["page_num"]) + " of " + str(total_pages) + ") Tj")
        stream_ops.append("ET")

        # Sections
        current_y = 690
        for sec_title, sec_body in page_data["sections"]:
            # Section Heading
            stream_ops.append("BT")
            stream_ops.append("/F2 11 Tf")
            stream_ops.append("0.1 0.2 0.4 rg")
            stream_ops.append(f"50 {current_y} Td")
            esc_sec_title = sec_title.replace("(", "\\(").replace(")", "\\)")
            stream_ops.append(f"({esc_sec_title}) Tj")
            stream_ops.append("ET")
            current_y -= 16

            # Section Body (word wrapped)
            words = sec_body.split()
            lines = []
            curr_line = []
            curr_len = 0
            max_len = 86  # approximate chars per line

            for word in words:
                if curr_len + len(word) + 1 > max_len:
                    lines.append(" ".join(curr_line))
                    curr_line = [word]
                    curr_len = len(word)
                else:
                    curr_line.append(word)
                    curr_len += len(word) + 1
            if curr_line:
                lines.append(" ".join(curr_line))

            stream_ops.append("BT")
            stream_ops.append("/F1 9.5 Tf")
            stream_ops.append("0.15 0.15 0.15 rg") # Dark grey text
            stream_ops.append(f"50 {current_y} Td")
            stream_ops.append("13 TL") # line leading

            for line_idx, line in enumerate(lines):
                esc_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                if line_idx == 0:
                    stream_ops.append(f"({esc_line}) Tj")
                else:
                    stream_ops.append(f"T* ({esc_line}) Tj")
            stream_ops.append("ET")

            current_y -= (len(lines) * 13 + 22)

        # Footer
        stream_ops.append("BT")
        stream_ops.append("/F1 8 Tf")
        stream_ops.append("0.5 0.5 0.5 rg")
        stream_ops.append(f"50 35 Td")
        stream_ops.append(f"(Smart Document Q&A Assistant Benchmark Dataset - Page {page_data['page_num']} of {total_pages} - Confidential & Proprietary) Tj")
        stream_ops.append("ET")

        stream_body = "\n".join(stream_ops)
        stream_bytes = stream_body.encode("latin-1")
        stream_len = len(stream_bytes)

        # Page object
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {stream_id} 0 R >>"
        )
        objects.append(page_dict)

        # Stream object
        stream_obj_str = (
            f"<< /Length {stream_len} >>\n"
            f"stream\n{stream_body}\nendstream"
        )
        objects.append(stream_obj_str)

    # Write complete PDF with xref table
    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []

    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("latin-1"))
        out.extend(obj.encode("latin-1"))
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(b"xref\n")
    out.extend(f"0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    out.extend(trailer.encode("latin-1"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(out)
    print(f"Successfully generated sample PDF at: {output_path} ({len(out)} bytes, {total_pages} pages)")


def ensure_sample_pdf_exists(output_path: str = SAMPLE_PDF_PATH) -> str:
    """Helper to ensure the sample PDF document is available."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        generate_pdf_pure_python(output_path)
    return output_path


if __name__ == "__main__":
    generate_pdf_pure_python(SAMPLE_PDF_PATH)

