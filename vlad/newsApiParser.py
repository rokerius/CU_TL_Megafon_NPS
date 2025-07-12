from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path

load_dotenv(dotenv_path="config/.env")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

to_date = datetime.now(timezone.utc)
from_date = to_date - timedelta(days=14)

from_str = from_date.strftime("%Y-%m-%d")
to_str = to_date.strftime("%Y-%m-%d")

url = "https://newsapi.org/v2/everything"
params = {
    "q": "МегаФон",
    "language": "ru",
    "sortBy": "publishedAt",
    "from": from_str,
    "to": to_str,
    "pageSize": 100,
    "page": 1,
    "apiKey": NEWS_API_KEY
}

all_articles = []

while True:
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print("Ошибка:", data.get("message"))
        break

    articles = data.get("articles", [])
    if not articles:
        break

    for article in articles:
        title = article["title"]
        source = article["source"]["name"]
        url_ = article["url"]
        published_at = article["publishedAt"]

        dt = datetime.fromisoformat(published_at.rstrip("Z"))
        formatted_date = dt.strftime("%d.%m.%Y %H:%M")

        all_articles.append([formatted_date, source, title, url_])

    print(f"Страница {params['page']} — загружено {len(articles)} статей.")
    if len(articles) < 100:
        break

    params["page"] += 1

csv_path = Path(__file__).parent / "data" / "megafon_news.csv"

with open(csv_path, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["date", "source", "title", "url"])
    writer.writerows(all_articles)

print(f"\n✅ Сохранено {len(all_articles)} статей в файл {csv_path}")