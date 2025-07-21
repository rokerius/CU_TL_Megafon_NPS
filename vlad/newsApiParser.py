from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
import feedparser

# Загрузка переменных окружения
load_dotenv(dotenv_path="config/.env")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Период новостей
to_date = datetime.now(timezone.utc)
from_date = to_date - timedelta(days=14)
from_str = from_date.strftime("%Y-%m-%d")
to_str = to_date.strftime("%Y-%m-%d")

# --- Парсинг из NewsAPI ---
print("🔎 Получение новостей через NewsAPI...")
newsapi_articles = []
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

while True:
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print("❌ Ошибка:", data.get("message"))
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

        newsapi_articles.append([formatted_date, source, title, url_])

    print(f"  ➕ Страница {params['page']} — загружено {len(articles)} статей.")
    if len(articles) < 100:
        break

    params["page"] += 1

# --- Парсинг из Google News RSS ---
print("🔎 Получение новостей через Google News RSS...")
google_news_articles = []
rss_url = "https://news.google.com/rss/search?q=МегаФон&hl=ru&gl=RU&ceid=RU:ru"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

try:
    response = requests.get(rss_url, headers=headers)
    feed = feedparser.parse(response.content)
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)

    for entry in feed.entries:
        title = entry.title
        link = entry.link
        source = entry.get("source", {}).get("title", "Google News")
        pub_date = entry.get("published_parsed")

        if not pub_date:
            continue

        dt = datetime(*pub_date[:6])
        if dt < cutoff:
            continue

        formatted_date = dt.strftime("%d.%m.%Y %H:%M")
        google_news_articles.append([formatted_date, source, title, link])

    print(f"  ➕ Загружено {len(google_news_articles)} статей из Google News RSS.")
except Exception as e:
    print("❌ Ошибка при парсинге Google News:", e)

# --- Сохраняем всё в CSV ---
all_articles = newsapi_articles + google_news_articles
csv_path = Path(__file__).parent / "data" / "megafon_news_combined.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["date", "source", "title", "url"])
    writer.writerows(all_articles)

print(f"\n✅ Всего сохранено {len(all_articles)} новостей в файл {csv_path}")
