from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

query = "МегаФон"
start_date = "1.1.2024"
end_date = "31.1.2024"
base_url = f"https://www.google.com/search?q={query}&hl=ru&tbm=nws&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--lang=ru-RU")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get(base_url)
time.sleep(2)

all_results = []

def parse_page():
    news_items = driver.find_elements(By.CSS_SELECTOR, "a.WlydOe")
    print(f"[DEBUG] На странице найдено новостных блоков: {len(news_items)}")
    for i, item in enumerate(news_items, 1):
        try:
            title = ""
            try:
                title = item.find_element(By.CSS_SELECTOR, "div.MBeuO").text.strip()
            except Exception:
                title = item.text.strip()
            url = item.get_attribute("href")
            # Попытка найти дату — обычно где-то поблизости
            try:
                parent = item.find_element(By.XPATH, './../..')
                date_span = parent.find_element(By.CSS_SELECTOR, "span.OSrXXb")
                date = date_span.text.strip()
            except Exception:
                date = ""
            print(f"[{i}] {title} ({date}) — {url}")
            all_results.append({'title': title, 'url': url, 'date': date})
        except Exception as e:
            print(f"[{i}] [ERROR] {e}")

page = 1
while True:
    print(f"[DEBUG] Парсим страницу {page}...")
    parse_page()
    # Поиск кнопки "Далее"
    try:
        next_btn = driver.find_element(By.ID, "pnnext")
        next_btn.click()
        time.sleep(2)
        page += 1
    except Exception:
        print("[DEBUG] Нет кнопки 'Далее' — это последняя страница.")
        break

driver.quit()

if all_results:
    with open("megafon_google_news.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "date"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[DEBUG] Сохранено {len(all_results)} новостей в megafon_google_news.csv")
else:
    print("[DEBUG] Новостей не найдено.")
