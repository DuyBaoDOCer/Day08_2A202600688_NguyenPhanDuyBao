"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # ------------------------------------------------------------------ #
    # Step 1: Chạy song song semantic search + lexical search             #
    # ------------------------------------------------------------------ #
    fetch_k = top_k * 3  # Lấy nhiều hơn để sau rerank còn đủ candidates

    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    try:
        dense_results = semantic_search(query, top_k=fetch_k)
    except Exception as e:
        print(f"  [!] Semantic search failed: {e}")

    try:
        sparse_results = lexical_search(query, top_k=fetch_k)
    except Exception as e:
        print(f"  [!] Lexical search failed: {e}")

    # ------------------------------------------------------------------ #
    # Step 2: Merge bằng Reciprocal Rank Fusion (RRF)                    #
    # ------------------------------------------------------------------ #
    all_lists = [lst for lst in [dense_results, sparse_results] if lst]
    if all_lists:
        merged = rerank_rrf(all_lists, top_k=fetch_k)
    else:
        merged = []

    # Gán source = "hybrid" cho tất cả kết quả sau merge
    for item in merged:
        item["source"] = "hybrid"

    # ------------------------------------------------------------------ #
    # Step 3: Rerank candidates                                           #
    # ------------------------------------------------------------------ #
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            # Đảm bảo source vẫn là "hybrid" sau rerank (rerank có thể copy metadata)
            for item in final_results:
                item.setdefault("source", "hybrid")
        except Exception as e:
            print(f"  [!] Reranking failed ({e}), using RRF-merged results.")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # ------------------------------------------------------------------ #
    # Step 4: Kiểm tra threshold → fallback sang PageIndex               #
    # ------------------------------------------------------------------ #
    best_score = final_results[0]["score"] if final_results else 0.0

    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid best score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback → PageIndex vectorless search."
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  [!] PageIndex fallback failed: {e}")
        # Nếu PageIndex cũng thất bại, trả về kết quả hybrid (dù thấp điểm)
        return final_results

    # ------------------------------------------------------------------ #
    # Step 5: Trả về top_k kết quả                                       #
    # ------------------------------------------------------------------ #
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
