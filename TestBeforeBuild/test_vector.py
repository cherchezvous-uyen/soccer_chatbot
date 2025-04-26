import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Bước 1: Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Bước 2: Tạo embedding function
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Bước 3: Kết nối tới Chroma
chroma_client = PersistentClient(path="../vectorstore/chroma")
collection = chroma_client.get_collection(
    name="sports_data",
    embedding_function=embedding_function
)

# Bước 4: Lấy toàn bộ tài liệu để hiển thị 5 cái đầu
results = collection.get(include=["documents", "metadatas"])
all_ids = results["ids"]

print(f"\n📊 Tổng số tài liệu trong collection: {collection.count()}")
print("📥 Hiển thị 5 tài liệu đầu tiên:\n")

for i in range(min(5, len(all_ids))):
    doc = results["documents"][i]
    meta = results["metadatas"][i]
    doc_id = all_ids[i]
    
    print(f"[{i+1}] 📄 ID: {doc_id}")
    print(f"📁 File gốc: {meta.get('filename', 'unknown')}")
    print("📝 Nội dung:")
    print(doc)
    print("-" * 60)

# Bước 5: Nhập ID cần truy vấn
target_id = input("\n🔍 Nhập ID tài liệu bạn muốn xem chi tiết: ").strip()

# Bước 6: Truy vấn theo ID vừa nhập
try:
    result = collection.get(
        ids=[target_id],
        include=["documents", "metadatas"]
    )

    if not result["documents"]:
        print("❌ Không tìm thấy tài liệu với ID đã nhập.")
    else:
        doc = result["documents"][0]
        meta = result["metadatas"][0]
        print(f"\n✅ Kết quả truy vấn cho ID: {target_id}")
        print(f"📁 File gốc: {meta.get('filename', 'unknown')}")
        print("📝 Nội dung:")
        print(doc)
except Exception as e:
    print(f"⚠️ Lỗi khi truy vấn: {e}")
