import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

INPUT_CSV = "megafon_rbc_news.csv"
OUTPUT_CSV = "megafon_rbc_news_with_positivity.csv"
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"  # ru sentiment: negative/neutral/positive
TEXT_FALLBACKS = ["text", "snippet", "desc", "description"]

# --- 1) читаем данные ---
print("[INFO] Читаем CSV:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print(f"[INFO] Загружено строк: {len(df)}")

def build_text(row):
    parts = [str(row.get("title", ""))]
    for c in TEXT_FALLBACKS:
        if c in row and isinstance(row[c], str) and row[c].strip():
            parts.append(row[c])
            break
    return ". ".join(p.strip() for p in parts if p and p.strip())

texts = df.apply(build_text, axis=1).fillna("").tolist()
print(f"[INFO] Сформировано текстов для анализа: {len(texts)}")
print("[DEBUG] Пример текста:", texts[0] if texts else "нет данных")

# --- 2) загружаем модель ---
print(f"[INFO] Загружаем модель: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    framework="pt",
    device=-1,            # CPU; если есть CUDA, поставь device=0
    truncation=True,
    max_length=256,
    return_all_scores=True
)
print("[INFO] Модель загружена успешно")

# --- 3) инференс батчами ---
BATCH = 32
positivity = []

print("[INFO] Начинаем инференс...")
for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    print(f"[DEBUG] Обработка батча {i//BATCH+1} ({i}-{i+len(batch)-1})")
    preds = pipe(batch)
    for idx, scores in enumerate(preds):
        by_label = {d["label"].lower(): float(d["score"]) for d in scores}
        p_pos = by_label.get("positive", 0.0)
        p_neg = by_label.get("negative", 0.0)
        score = round(p_pos - p_neg, 4)
        positivity.append(score)

        if idx == 0:  # показать пример внутри батча
            print(f"[DEBUG] Пример предсказания: {by_label}, итог={score}")

# --- 4) сохраняем ---
df["positivity"] = positivity
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"[INFO] Готово: {os.path.abspath(OUTPUT_CSV)}")
print("[INFO] Первые 5 строк с результатами:")
print(df[["title", "positivity"]].head())