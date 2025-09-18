from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
import csv
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options

query = "МегаФон"
global_start = datetime(2024, 1, 1)
global_end = datetime.now()
step_days = 7

month_map = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12',
}

def parse_rbc_date(date_text):
    try:
        parts = date_text.strip().split(", ")
        # Последние два элемента — дата и время
        date_str = parts[-2]  # например: '14 фев 2024'
        time_str = parts[-1]  # например: '14:39'

        day, mon_rus, year = date_str.split()
        mon_num = month_map.get(mon_rus.lower())
        if not mon_num:
            return None
        full_str = f"{day}.{mon_num}.{year} {time_str}"
        return datetime.strptime(full_str, "%d.%m.%Y %H:%M")
    except Exception:
        return None

def format_date(d: datetime):
    return d.strftime("%d.%m.%Y")

options = Options()
# options.add_argument("--headless")  # для macOS можно оставить окно
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--lang=ru-RU")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.7204.102 Safari/537.36")

print("[INFO] Запускаем браузер...")
driver = uc.Chrome(options=options)
print("[INFO] Браузер успешно запущен!")

all_articles = []
visited_urls = set()

current_start = global_start
while current_start < global_end:
    current_end = min(current_start + timedelta(days=step_days - 1), global_end)
    date_from = format_date(current_start)
    date_to = format_date(current_end)
    print(f"[INFO] Обрабатываем: {date_from} — {date_to}")

    articles_collected = 0
    page = 1
    while articles_collected < 20:
        url = (
            f"https://www.rbc.ru/search/?query={query}"
            f"&dateFrom={date_from}&dateTo={date_to}&page={page}"
        )
        driver.get(url)
        time.sleep(4)

        articles = driver.find_elements(By.CSS_SELECTOR, "a.search-item__link")
        if not articles:
            print("[INFO] Статей не найдено.")
            break

        for a in articles:
            try:
                title = a.find_element(By.CSS_SELECTOR, "span.search-item__title").text.strip()
                url = a.get_attribute("href")
                if url in visited_urls:
                    continue
                visited_urls.add(url)

                # Парсим дату
                try:
                    wrap = a.find_element(By.XPATH, ".//ancestor::div[contains(@class, 'search-item__wrap')]")
                    date_text = wrap.find_element(By.CSS_SELECTOR, "span.search-item__category").text.strip()
                    parsed_date = parse_rbc_date(date_text)
                except Exception:
                    parsed_date = None

                date_final = parsed_date.strftime("%d.%m.%Y %H:%M") if parsed_date else f"{date_from} ~"

                print(f"[ARTICLE] {title} — {url} ({date_final})")
                all_articles.append({"title": title, "url": url, "date": date_final})
                articles_collected += 1

                if articles_collected >= 20:
                    break
            except Exception as e:
                print(f"[ERROR] {e}")
        if len(articles) < 20:
            break
        page += 1

    current_start += timedelta(days=step_days)

driver.quit()

if all_articles:
    with open("megafon_rbc_news_filtered.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "date"])
        writer.writeheader()
        writer.writerows(all_articles)
    import os
    print(f"[INFO] Сохранено {len(all_articles)} новостей в: {os.path.abspath('megafon_rbc_news_filtered.csv')}")
else:
    print("[INFO] Новостей не найдено.")