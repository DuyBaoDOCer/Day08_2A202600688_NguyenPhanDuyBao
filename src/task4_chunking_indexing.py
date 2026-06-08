"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy và giải thích vì sao
# CHUNK_SIZE = 500: Đủ nhỏ để giữ ngữ cảnh tập trung vào một vấn đề pháp lý hoặc bài viết cụ thể, không làm loãng vector đại diện.
# CHUNK_OVERLAP = 50: Đảm bảo không mất mát thông tin liên kết giữa các ranh giới cắt của các chunks.
# CHUNKING_METHOD = "recursive": Sử dụng RecursiveCharacterTextSplitter để tách theo cấu trúc tự nhiên (dòng, đoạn) của tài liệu tiếng Việt.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Chọn embedding model và giải thích
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2": Lựa chọn mô hình siêu nhẹ (khoảng 90MB), tải nhanh và tiết kiệm tài nguyên.
# EMBEDDING_DIM = 384: Số chiều vector tương ứng của mô hình all-MiniLM-L6-v2.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chọn vector store
# VECTOR_STORE = "weaviate": Sử dụng Weaviate làm Vector Database chính thức theo khuyến nghị.
# Hỗ trợ tìm kiếm Hybrid (dense + BM25) tích hợp sẵn, hiệu năng cao và khả năng mở rộng tốt.
# Hỗ trợ cả Weaviate Cloud (WCD) qua biến môi trường (.env) và local instance qua Docker.
VECTOR_STORE = "weaviate"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file():
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file.parent) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (Weaviate).
    """
    import os
    import weaviate
    from weaviate.classes.config import Property, DataType, Configure
    from dotenv import load_dotenv
    
    load_dotenv()
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    
    # Kết nối Weaviate (Cloud nếu có cấu hình trong .env, ngược lại kết nối Local)
    try:
        if weaviate_url and weaviate_key:
            print(f"Connecting to Weaviate Cloud at: {weaviate_url}")
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_key)
            )
        else:
            print("Connecting to local Weaviate instance...")
            client = weaviate.connect_to_local()
    except Exception as e:
        print(f"\n[!] Lỗi kết nối đến Weaviate: {e}")
        print("Vui lòng kiểm tra xem:")
        print("  1. Docker Weaviate đã chạy (nếu dùng local).")
        print("  2. Hoặc WEAVIATE_URL và WEAVIATE_API_KEY đã được điền chính xác trong file .env (nếu dùng Cloud).\n")
        raise e

    try:
        # Tên bộ sưu tập tài liệu trong Weaviate
        class_name = "DrugLawDocs"
        
        # Xóa collection cũ nếu đã tồn tại để tránh trùng lặp dữ liệu khi index lại
        if client.collections.exists(class_name):
            print(f"Collection '{class_name}' already exists. Recreating...")
            client.collections.delete(class_name)
            
        # Tạo collection mới với định nghĩa các thuộc tính
        collection = client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),  # Chúng ta truyền vector thủ công
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        
        # Insert các chunks thông qua cơ chế Dynamic Batching của Weaviate v4
        print(f"Indexing {len(chunks)} chunks to Weaviate...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
        
        # Kiểm tra lỗi batch
        if collection.batch.failed_objects:
            print(f"  [!] Có {len(collection.batch.failed_objects)} đối tượng gặp lỗi khi lưu vào database.")
        else:
            print(f"✓ Hoàn thành lưu trữ dữ liệu vào Weaviate '{class_name}'.")
            
    finally:
        client.close()
        print("Weaviate connection closed.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
