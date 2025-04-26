import json

def normalize_player(raw):
    return {
        "player_id": raw["player"]["id"],
        "player_name": raw["player"]["name"].lower(),
        "firstname": raw["player"]["firstname"].lower(),
        "lastname": raw["player"]["lastname"].lower(),
        "age": raw["player"]["age"],
        "nationality": raw["player"]["nationality"].lower(),
        "photo_url": raw["player"]["photo"],
        "team_id": raw["team"]["id"],
        "team_name": raw["team"]["name"].lower(),
        "team_logo_url": raw["team"]["logo"],
        "appearances": raw["games"]["appearences"],
        "position": raw["games"]["position"].lower(),
        "goals_total": raw["goals"]["total"]
    }

# Đọc file JSON gốc
with open("../json/TopScorers_Filtered.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Nếu là danh sách, thì duyệt qua từng phần tử
if isinstance(raw_data, list):
    cleaned_TopScorers = [normalize_player(player) for player in raw_data]
else:
    cleaned_TopScorers = [normalize_player(raw_data)]  # Trường hợp chỉ có 1 cầu thủ

# Ghi ra file JSON đã làm sạch
with open("cleaned_TopScorers.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_TopScorers, f, indent=2, ensure_ascii=False)

print("✅ Đã làm sạch và lưu vào cleaned_TopScorers.json")
