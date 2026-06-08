"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve chunks (Task 9)
    2. Reorder chunks (Lost-in-the-Middle mitigation)
    3. Format context với source labels
    4. Inject vào prompt với SYSTEM_PROMPT
    5. Call OpenAI gpt-4o-mini
    6. Return answer có citation + sources metadata

Lý do chọn tham số:
    - TOP_K = 5    : Đủ evidence từ nhiều nguồn, không quá dài (tránh context overflow
                     và lost-in-the-middle khi quá nhiều chunks)
    - TOP_P = 0.9  : Nucleus sampling — giữ 90% xác suất tích luỹ. Đủ đa dạng từ ngữ
                     mà không quá random (tránh hallucination)
    - TEMPERATURE  : 0.3 — RAG cần output factual, ít sáng tạo
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random, phù hợp RAG factual
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần faithful, ít sáng tạo, không hallucinate
TEMPERATURE = 0.3

# Model
LLM_MODEL = "gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Thuật toán — interleave từ hai đầu của danh sách đã sort:
        Input (sorted by score desc): [A, B, C, D, E]  (A = best, E = worst)
        Output:                       [A, C, E, D, B]
          - Vị trí đầu  (index 0): A — best
          - Vị trí cuối (index -1): B — second best
          - Giữa: C, D, E — kém quan trọng hơn

    Ví dụ với 5 chunks:
        sorted_chunks = [chunk0(1.0), chunk1(0.9), chunk2(0.8), chunk3(0.7), chunk4(0.6)]
        odd_indices   = [0, 2, 4]  → [chunk0, chunk2, chunk4]  đặt ở đầu
        even_indices  = [1, 3]     → [chunk1, chunk3] đảo ngược → [chunk3, chunk1] đặt cuối
        result        = [chunk0, chunk2, chunk4, chunk3, chunk1]

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention cho chunks quan trọng nhất.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # Chia thành hai nhóm: odd-index (đặt trước) và even-index (đặt sau, đảo)
    # Assumption: input đã được sort by score descending (chunk[0] = best score)
    front = chunks[::2]          # index 0, 2, 4, ... → chứa chunk best + mid
    back  = chunks[1::2][::-1]  # index 1, 3, 5, ... → đảo ngược → chunk 2nd-best cuối cùng

    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite chính xác.

    Format mỗi chunk:
        [Document {i} | Source: {source} | Type: {type}]
        {content}

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string, ngăn cách bởi '---'.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", metadata.get("doc_type", "unknown"))

        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}"
        )

    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9: semantic + lexical + rerank + fallback)
        2. Reorder chunks để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call gpt-4o-mini
        6. Return {'answer', 'sources', 'retrieval_source'}

    Args:
        query:  Câu hỏi của user
        top_k:  Số chunks context (default 5)

    Returns:
        {
            'answer': str,           # Câu trả lời tiếng Việt có citation [Nguồn, Năm]
            'sources': list[dict],   # Các chunks đã dùng làm context
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # ------------------------------------------------------------------
    # Step 1: Retrieve relevant chunks từ pipeline (Task 9)
    # ------------------------------------------------------------------
    chunks = retrieve(query, top_k=top_k)

    # Nếu không có chunks nào → trả về thông báo không đủ evidence
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none"
        }

    # ------------------------------------------------------------------
    # Step 2: Reorder để tránh lost in the middle
    # ------------------------------------------------------------------
    reordered = reorder_for_llm(chunks)

    # ------------------------------------------------------------------
    # Step 3: Format context với source labels để LLM cite được
    # ------------------------------------------------------------------
    context = format_context(reordered)

    # ------------------------------------------------------------------
    # Step 4: Build prompt
    #   - SYSTEM_PROMPT: yêu cầu citation + không hallucinate
    #   - user_message: context đầy đủ + câu hỏi
    # ------------------------------------------------------------------
    user_message = (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    # ------------------------------------------------------------------
    # Step 5: Call LLM — gpt-4o-mini
    #   - temperature=0.3: RAG cần factual, ít sáng tạo
    #   - top_p=0.9: nucleus sampling, không quá random
    # ------------------------------------------------------------------
    from openai import OpenAI

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env file.")

    # Automatically detect if key is an NVIDIA API key
    if openai_key.startswith("nvapi-"):
        base_url = "https://integrate.api.nvidia.com/v1"
        model = "openai/gpt-oss-20b"
    else:
        base_url = None
        model = LLM_MODEL

    client = OpenAI(api_key=openai_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # ------------------------------------------------------------------
    # Step 6: Return kết quả
    # ------------------------------------------------------------------
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
