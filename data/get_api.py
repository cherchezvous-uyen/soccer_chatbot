import requests
import json

url = 'https://api.football-data.org/v4/persons/2019/matches?status=FINISHED'
headers = {
    'X-Auth-Token': '51d1b38fad36412b97e727ac97413bd7'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()

    # Lưu toàn bộ dữ liệu vào file JSON
    with open('aa.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✅ Đã lưu toàn bộ dữ liệu bảng xếp hạng Premier League vào aa.json")
else:
    print(f"❌ Lỗi khi gọi API: {response.status_code}")
