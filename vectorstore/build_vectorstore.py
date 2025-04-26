import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Bước 1: Load API Key từ .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise Exception("Không tìm thấy OPENAI_API_KEY trong file .env!")

# Bước 2: Tạo embedding function
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Bước 3: Kết nối Chroma vectorstore kiểu mới
chroma_client = chromadb.PersistentClient(path="vectorstore\chroma")

# Bước 4: Tạo hoặc lấy collection
collection = chroma_client.get_or_create_collection(
    name="sports_data",
    embedding_function=embedding_function
)

data_dir = 'data\ChunkedData'# Cập nhật đường dẫn tương đối
doc_id = 0
documents = []
ids = []
metadatas = []

if not os.path.exists(data_dir):
    raise Exception(f"Thư mục {data_dir} không tồn tại!")

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                if isinstance(chunks, str):
                    chunks = [json.loads(chunk) for chunk in chunks]
        except Exception as e:
            print(f"❌ Lỗi khi đọc {filename}: {e}")
            continue

        for chunk in tqdm(chunks, desc=f"Embedding {filename}"):
            try:
                content = json.dumps(chunk, ensure_ascii=False) if isinstance(chunk, dict) else str(chunk)
                documents.append(content)
                ids.append(f"doc_{doc_id}")
                metadatas.append({"filename": filename})
                doc_id += 1
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý đoạn trong {filename}: {e}")

# Bước 6: Thêm dữ liệu vào collection
if documents:
    try:
        collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"\n✅ Đã thêm {doc_id} documents vào collection 'sports_data'.")
    except Exception as e:
        print(f"❌ Lỗi khi thêm dữ liệu vào collection: {e}")
else:
    print("⚠️ Không có dữ liệu nào để thêm vào collection.")

# Bước 7: Kiểm tra collection
print(f"📊 Số lượng tài liệu trong collection: {collection.count()}")