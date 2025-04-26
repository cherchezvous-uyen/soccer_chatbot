

# test2
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os, time
import json
import traceback

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB
base_path = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.abspath(os.path.join(base_path, "../vectorstore/chroma"))
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection = chroma_client.get_collection("sports_data")

# Lưu câu hỏi và câu trả lời vào file JSON
def save_to_json(question, answer, filename="chat_history.json"):
    try:
        # Đọc dữ liệu hiện tại trong file (nếu có)
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # Thêm câu hỏi và câu trả lời vào danh sách
        data.append({"question": question, "answer": answer})

        # Ghi dữ liệu vào file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Đã lưu câu hỏi và câu trả lời vào {filename}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào file: {str(e)}")

def get_embedding(query):
    return openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

def hybrid_search_rerank(query, top_k=5):
    if collection.count() == 0:
        return []
    
    # Hiển thị số lượng documents và collections trong ChromaDB
    num_documents = collection.count()
    print(f"Số lượng documents trong collection: {num_documents}")
    
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        query_texts=[query],
        n_results=top_k
    )
    
    docs = zip(results['ids'][0], results['documents'][0], results['distances'][0])
    
    return sorted(docs, key=lambda x: x[2])

def generate_answer(query, reranked_docs, history=None):
    if not reranked_docs:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    context = "\n---\n".join([doc for _, doc, _ in reranked_docs])
    
    # Hiển thị context (dữ liệu được sử dụng để trả lời câu hỏi)
    print(f"Context tìm thấy từ dữ liệu:\n{context}")

    system_prompt = """
    Bạn là một chuyên giagia trả lời câu hỏi chỉ dựa trên dữ liệu được cung cấp.
    Bạn phải biết các trả lời chào hỏi cơ bản như 1 con người thực sự.
    Bạn hãy trả lời theo phong cách vui vẻ.
    - KHÔNG sử dụng kiến thức nền
    - Hãy duy trì hội thoại liên tục. Nếu người dùng dùng từ như "he", "his", "that player", "anh ấy", "anh ta", "mùa giải đó", "giải đấu trên", v.v.  hãy cố gắng hiểu theo ngữ cảnh từ câu trước đó.
    - Nếu không có dữ liệu, trả lời "Không đủ thông tin trong cơ sở dữ liệu."
    """
    messages = [{"role": "system", "content": system_prompt.strip()}]
    if history:
        messages += history[-10:]

    user_prompt = f"""
    Dưới đây là dữ liệu:

    {context}

    Trả lời câu hỏi: {query}
    """
    messages.append({"role": "user", "content": user_prompt.strip()})

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def handle_query(query, history=None):
    reranked_docs = hybrid_search_rerank(query)
    return generate_answer(query, reranked_docs, history)

#test
if __name__ == "__main__":
    try:
        start = time.time()
        query = input("Nhap cau hoi:")
        hyb_docs = hybrid_search_rerank(query)
        print(f"\n Sắp xếp theo độ tương đồng tài liệu")
        for i, (ids,docs, distance) in enumerate (hyb_docs,1):
            tag = "tot" if distance < 0.3 else ("khong tot" if distance < 0.6 else "xau") 
            print(f"ID: [{ids}], , Khoảng cách: {distance:.4f},tag, Tài liệu: {docs[:100]}")
        print(f"Thời gian tìm kiếm: {time.time() - start:.2f} giây")
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {str(e)}")
        traceback.print_exc()
