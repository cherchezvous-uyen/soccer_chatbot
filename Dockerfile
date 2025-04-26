# Sử dụng Python 3.11 làm base image
FROM python:3.11

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements.txt vào container
COPY requirements.txt /app/

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt Gunicorn
RUN pip install gunicorn

# Sao chép mã nguồn của bạn vào container
COPY . /app/

# Mở cổng 8000 (hoặc cổng bạn sử dụng)
EXPOSE 8000

# Chạy ứng dụng bằng Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
