# 🤖 Chatbot Bóng Đá với OpenAI + RAG

## Hướng dẫn chạy

LÀM THEO TƯNG BƯỚC DƯỚI ĐỂ CHẠY PROJECT
1. Tải build tools về, LƯU Ý KHI SETUP chọn Desktop development with C++, LINK TẢI Ở DƯỚI DƯỚI
https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. pip install -r requirements.txt
3. pip install -U langchain-openai
4. pip install -U langchain-community
5. python vectorstore/build_vectorstore.py
6. python app.py
7. click vào port trên terminal khi chạy app.py thành công để mở giao diện

import thêm hàm dịch Tiếng Việt Tiếng ANh
pip install deep-translator
chạy app bị lỗi httpx, core blabla thì 
# pip install --upgrade chromadb urllib3 httpx
