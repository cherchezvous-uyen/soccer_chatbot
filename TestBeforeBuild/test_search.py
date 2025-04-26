from chromadb import PersistentClient
import openai
import os
from dotenv import load_dotenv
import time

# Load API Key từ file .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Kết nối tới Chroma client
chroma_client = PersistentClient(path="../vectorstore/chroma")

# Truy cập vào collection "sports_data"
collection = chroma_client.get_collection("sports_data")  # Đảm bảo "sports_data" là tên collection đúng

# Cập nhật OpenAIEmbeddingFunction với API key và mô hình phù hợp
openai.api_key = openai_key

# Hàm lấy embedding từ OpenAI API
def get_embedding_from_openai(query):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # model với embedding kích thước 1536
        input=query
    )
    return response.data[0].embedding  # Truy cập bằng thuộc tính data

def normal_search(query, collection, top_k=10):
    # Lấy embedding của query từ OpenAI
    query_embedding = get_embedding_from_openai(query)
    
    # Tìm kiếm trong collection với embedding
    text_results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return text_results

def hybrid_search_rerank(query, collection, top_k=10):
    # Lấy embedding của query từ OpenAI
    query_embedding = get_embedding_from_openai(query)

    # Tìm kiếm embedding
    embed_results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # Tìm kiếm text (nếu cần thiết, đảm bảo text search yêu cầu embedding tương tự)
    text_results = collection.query(query_embeddings=[query_embedding], n_results=top_k)  # Cập nhật text search thành query_embeddings

    docs = embed_results['documents'][0] + text_results['documents'][0]
    ids = embed_results['ids'][0] + text_results['ids'][0]
    distances = embed_results['distances'][0] + text_results['distances'][0]

    # Sắp xếp theo khoảng cách từ gần nhất
    ranked = sorted(zip(ids, docs, distances), key=lambda x: x[2])

    seen = set()
    reranked_results = []
    for doc_id, doc, dist in ranked:
        if doc_id not in seen:
            seen.add(doc_id)
            reranked_results.append((doc_id, doc, dist))
        if len(reranked_results) >= top_k:
            break

    return reranked_results

# Test thời gian cho search bình thường
query = "Ai là người ghi nhiều bàn thắng nhất gần đây"  # Thay câu truy vấn của em vào đây

# Bước 1: Test thời gian cho tìm kiếm bình thường
start_time = time.time()
normal_results = normal_search(query, collection)
normal_search_time = time.time() - start_time
print(f"Thời gian tìm kiếm bình thường: {normal_search_time:.4f} giây \n")
# In kết quả tìm kiếm bình thường
print("\n🔍 Kết quả tìm kiếm bình thường:")
for i, doc in enumerate(normal_results['documents'][0]):
    print(f"{i+1}. {doc}")

# Bước 2: Test thời gian cho Hybrid Search + Reranking
start_time = time.time()
hybrid_results = hybrid_search_rerank(query, collection)
hybrid_search_time = time.time() - start_time
print(f"\nThời gian tìm kiếm với Hybrid + Reranking: {hybrid_search_time:.4f} giây")
# In kết quả tìm kiếm Hybrid + Reranking
print("\n🤖 Kết quả tìm kiếm Hybrid + Reranking:")
for i, (doc_id, doc, dist) in enumerate(hybrid_results):
    print(f"{i+1}. {doc} (ID: {doc_id}, Distance: {dist:.4f})")