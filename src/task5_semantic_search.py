"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # 1. Embed query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # 2. Connect to Weaviate (Cloud if configured, otherwise Local)
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
        collection = client.collections.get("DrugLawDocs")
        
        # Truy vấn vector gần nhất
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        search_results = []
        for obj in results.objects:
            # Weaviate lưu khoảng cách (distance). 
            # Với khoảng cách Cosine, độ tương đồng (similarity score) = 1.0 - distance
            distance = obj.metadata.distance if obj.metadata.distance is not None else 0.0
            score = 1.0 - distance
            
            search_results.append({
                "content": obj.properties.get("content", ""),
                "score": float(score),
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0)
                }
            })
            
        # Sắp xếp kết quả giảm dần theo score
        search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results
        
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
