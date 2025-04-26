from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import json
import os
import fitz

print("Bắt đầu tiền xử lý JSON, JSONL, PDF và TXT...")

data_dir = "../data/CleanJSON"

# JSON
def create_documents_from_json(data, source_file):
    docs = []
    for item in data:
        text = json.dumps(item, ensure_ascii=False, indent=2)
        metadata = {"source": source_file}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

# PDF
def create_documents_from_pdf(file_path):
    docs = []
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            metadata = {"source": file_path, "page": page_num}
            docs.append(Document(page_content=text.strip(), metadata=metadata))
    return docs

# JSONL
def create_documents_from_jsonl(file_path):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    text = json.dumps(item, ensure_ascii=False, indent=2)
                    metadata = {"source": file_path}
                    docs.append(Document(page_content=text, metadata=metadata))
                except json.JSONDecodeError:
                    print(f"Lỗi JSONL ở dòng: {line}")
    return docs

# TXT
def create_documents_from_txt(file_path):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        if text.strip():
            metadata = {"source": file_path}
            docs.append(Document(page_content=text.strip(), metadata=metadata))
    return docs

# Chunk documents
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_documents = splitter.split_documents(documents)
    
    # Đánh dấu mỗi chunk bằng metadata
    for idx, chunk in enumerate(chunked_documents):
        chunk.metadata["chunk_index"] = idx + 1  # Đánh dấu chunk thứ mấy

    return chunked_documents

# Process all files
def process_files_in_directory(data_dir):
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.endswith(".json") and not filename.endswith(".jsonl"):
            print(f"🟡 JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = create_documents_from_json(data, file_path)

        elif filename.endswith(".jsonl"):
            print(f"🟣 JSONL: {file_path}")
            documents = create_documents_from_jsonl(file_path)

        elif filename.endswith(".pdf"):
            print(f"🔵 PDF: {file_path}")
            documents = create_documents_from_pdf(file_path)

        elif filename.endswith(".txt"):
            print(f"🟢 TXT: {file_path}")
            documents = create_documents_from_txt(file_path)

        else:
            continue

        # Chunk
        chunked_docs = chunk_documents(documents)

        # In ra kết quả
        print(f"\n--- Kết quả trước khi chunking ---")
        for doc in documents[:3]:  # In 3 document đầu tiên từ mỗi loại file
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content[:500]}...")  # Chỉ in phần đầu tiên

        print(f"\n--- Kết quả sau khi chunking ---")
        for chunk in chunked_docs[:3]:  # In 3 chunk đầu tiên từ kết quả chunking
            print(f"Chunk Index: {chunk.metadata['chunk_index']}")
            print(f"Chunk Content: {chunk.page_content[:500]}...")  # Chỉ in phần đầu tiên

        print(f"\n{'-'*50}\n")

# Chạy
process_files_in_directory(data_dir)
