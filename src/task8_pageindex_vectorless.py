"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex import PageIndexClient
    
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY is not configured in .env file.")
        return
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Quét toàn bộ file markdown trong thư mục standardized để upload
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file():
            print(f"Uploading to PageIndex: {md_file.name}...")
            try:
                result = client.submit_document(file_path=str(md_file))
                doc_id = result.get("doc_id")
                print(f"  ✓ Uploaded: {md_file.name} (doc_id: {doc_id})")
            except Exception as e:
                print(f"  [!] Failed to upload {md_file.name}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient
    
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is not configured in .env file.")
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Liệt kê danh sách tài liệu hiện có trên PageIndex
    doc_res = client.list_documents()
    docs = doc_res.get("documents", [])
    
    if not docs:
        print("⚠ Không tìm thấy tài liệu nào trên tài khoản PageIndex của bạn.")
        return []
        
    doc_ids = [d.get("id") or d.get("doc_id") for d in docs if d.get("id") or d.get("doc_id")]
    
    # Thực hiện truy vấn thông qua chat_completions giới hạn phạm vi tài liệu (reasoning-based)
    messages = [{"role": "user", "content": query}]
    response = client.chat_completions(
        messages=messages,
        doc_id=doc_ids,
        enable_citations=True
    )
    
    answer = response["choices"][0]["message"]["content"]
    
    return [
        {
            "content": answer,
            "score": 1.0,
            "metadata": {
                "doc_ids": doc_ids,
                "note": "PageIndex vectorless RAG reasoning output"
            },
            "source": "pageindex"
        }
    ]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
