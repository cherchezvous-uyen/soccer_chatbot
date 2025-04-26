from flask import Flask, request, jsonify, render_template
from retrieval.rag import handle_query  # Chỉ cần sử dụng hàm handle_query
from dotenv import load_dotenv
import traceback

load_dotenv()

app = Flask(__name__)
chat_history = []

@app.route('/')
def home():
    return render_template("index.html")  # giao diện HTML tĩnh

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'response': '⚠️ Bạn chưa nhập câu hỏi nào!'}), 400

    try:
        # Lưu câu hỏi vào history
        chat_history.append({"role": "user", "content": message})

        # Gọi xử lý chính
        answer = handle_query(message, chat_history)

        # Lưu câu trả lời vào history
        chat_history.append({"role": "assistant", "content": answer})

        # Giới hạn lịch sử hội thoại
        if len(chat_history) > 20:
            chat_history[:] = chat_history[-20:]

        return jsonify({'response': answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f'🚨 Lỗi server: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

