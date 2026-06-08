"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# CORPUS và BM25_INDEX sẽ được khởi tạo lười (lazy loading) khi chạy tìm kiếm lần đầu tiên
_CORPUS: list[dict] = []
_BM25_INDEX = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    # Tokenize đơn giản bằng cách tách từ và chuyển thành viết thường (lower)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def get_bm25_index_and_corpus():
    """
    Khởi tạo và lấy BM25 Index cùng Corpus từ Weaviate.
    """
    global _CORPUS, _BM25_INDEX
    if _BM25_INDEX is not None:
        return _BM25_INDEX, _CORPUS

    import os
    import weaviate
    from dotenv import load_dotenv
    load_dotenv()

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_key:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_key)
        )
    else:
        client = weaviate.connect_to_local()

    try:
        class_name = "DrugLawDocs"
        collection = client.collections.get(class_name)
        response = collection.query.fetch_objects(limit=10000)
        _CORPUS = [
            {
                "content": obj.properties.get("content", ""),
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0)
                }
            }
            for obj in response.objects
        ]
        print(f"✓ Loaded {len(_CORPUS)} chunks from Weaviate collection '{class_name}'.")
    finally:
        client.close()

    # Xây dựng BM25 Index
    _BM25_INDEX = build_bm25_index(_CORPUS)
    return _BM25_INDEX, _CORPUS


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    # Lấy BM25 index và corpus
    bm25, corpus = get_bm25_index_and_corpus()

    # Tokenize truy vấn
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k chỉ mục có điểm số cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"]
        })

    # Đảm bảo được sắp xếp giảm dần theo điểm số
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
