#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, argparse, hashlib, logging
from io import BytesIO
from typing import Any, Optional, Iterable, List
from datetime import date, datetime

import pandas as pd
import requests
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("enrich_net")

# ----------------------------- #
# JSON-safe cache
# ----------------------------- #
CACHE_DIR = ".cache_macros"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def _json_safe(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, pd.DataFrame):
        d = x.copy()
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.strftime("%Y-%m-%d")
        return d.to_dict(orient="list")
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)

def cache_get(key: str):
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_set(key: str, data):
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Кэш не сохранился: {e}")

# ----------------------------- #
# helpers
# ----------------------------- #
def to_month(val, dayfirst=False) -> Optional[pd.Timestamp]:
    if isinstance(val, str):
        dt = pd.to_datetime(val, errors="coerce", dayfirst=dayfirst)
    else:
        dt = pd.to_datetime(val, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt.year, dt.month, 1)

def month_span(series: pd.Series):
    s = series.dropna()
    return (s.min(), s.max()) if not s.empty else (None, None)

def get_html_tables(url: str, verify=True, retries=3, sleep=1.5) -> List[pd.DataFrame]:
    key = f"HTML::{url}::{int(verify)}"
    c = cache_get(key)
    if c:
        dfs = [pd.DataFrame(tbl) for tbl in c]
        for df in dfs:
            for col in df.columns:
                if df[col].dtype == object and str(df[col].iloc[:1].values[0]).startswith("dt:"):
                    df[col] = pd.to_datetime(df[col].str[3:], errors="coerce")
        return dfs
    for a in range(retries):
        try:
            log.info(f"GET {url}")
            r = requests.get(url, timeout=40, verify=verify)
            r.raise_for_status()
            dfs = pd.read_html(BytesIO(r.content), decimal=",", thousands=" ")
            cache_set(key, dfs)
            return dfs
        except Exception as e:
            log.warning(f"read_html fail {url}: {e}; retry...")
            time.sleep(sleep)
    return []

def cbr_xml_daily(d: date, verify=True, retries=3, sleep=1.0) -> dict:
    """
    https://www.cbr.ru/scripts/XML_daily.asp?date_req=DD/MM/YYYY
    Возвращает словарь { 'USD': 90.1, 'EUR': 98.2, ... }.
    """
    key = f"CBR_XML_DAILY::{d.isoformat()}::{int(verify)}"
    c = cache_get(key)
    if c: return c
    url = "https://www.cbr.ru/scripts/XML_daily.asp"
    params = {"date_req": d.strftime("%d/%m/%Y")}
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20, verify=verify)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            vals = {}
            for v in root.findall("Valute"):
                code = v.findtext("CharCode", "").strip()
                nominal = float(v.findtext("Nominal", "1").replace(",", "."))
                value = float(v.findtext("Value", "0").replace(",", "."))
                if nominal:
                    vals[code] = value / nominal
            cache_set(key, vals)
            return vals
        except Exception as e:
            log.warning(f"XML_daily {params['date_req']}: {e}; retry...")
            time.sleep(sleep)
    return {}

def fx_monthly_avg(months: Iterable[pd.Timestamp], codes=("USD","EUR","CNY"), verify=True) -> pd.DataFrame:
    out = []
    for m in sorted(set(months)):
        start = pd.Timestamp(m.year, m.month, 1)
        end = (start + pd.offsets.MonthBegin(1))
        cur = start
        acc = {c: [] for c in codes}
        while cur < end:
            di = cbr_xml_daily(cur.date(), verify=verify)
            for c in codes:
                if c in di: acc[c].append(di[c])
            cur += pd.Timedelta(days=1)
        rec = {"month": start}
        for c in codes:
            rec[c+"_avg_month_cbr"] = round(sum(acc[c]) / len(acc[c]), 6) if acc[c] else None
        out.append(rec)
    return pd.DataFrame(out)

# ----------------------------- #
# FIN: RUONIA, MOSPRIME, IMOEX
# ----------------------------- #
def fetch_ruonia_month_avg(verify=True) -> pd.DataFrame:
    url = "https://www.cbr.ru/hd_base/ruonia/"
    key = f"RUONIA::{int(verify)}"
    c = cache_get(key)
    if c:
        df = pd.DataFrame(c); df["month"] = pd.to_datetime(df["month"]); return df

    dfs = get_html_tables(url, verify=verify)
    if not dfs: return pd.DataFrame(columns=["month","ruonia_avg_month_pct"])

    t = None
    for d in dfs:
        cols = [str(x).lower() for x in d.columns]
        if any("дата" in x for x in cols) and any(("ruonia" in x) or ("руония" in x) for x in cols):
            t = d; break
    if t is None: t = dfs[0]

    t.columns = [str(c).strip() for c in t.columns]
    date_col = next((c for c in t.columns if "ата" in c or "date" in c.lower()), t.columns[0])
    rate_col = next((c for c in t.columns if "ruonia" in c.lower() or "руония" in c.lower()), t.columns[-1])

    t = t[[date_col, rate_col]].rename(columns={date_col:"date", rate_col:"ruonia"})
    t["date"] = pd.to_datetime(t["date"], dayfirst=True, errors="coerce")
    t["ruonia"] = pd.to_numeric(t["ruonia"].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    t.dropna(subset=["date","ruonia"], inplace=True)
    t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
    out = t.groupby("month", as_index=False).agg(ruonia_avg_month_pct=("ruonia","mean"))
    out["ruonia_avg_month_pct"] = out["ruonia_avg_month_pct"].round(4)
    cache_set(key, out)
    return out

def fetch_mosprime_month_last(verify=True) -> pd.DataFrame:
    url = "https://www.cbr.ru/hd_base/mosprime/"
    key = f"MOSPRIME::{int(verify)}"
    c = cache_get(key)
    if c:
        df = pd.DataFrame(c); df["month"] = pd.to_datetime(df["month"]); return df

    dfs = get_html_tables(url, verify=verify)
    if not dfs: return pd.DataFrame(columns=["month","mosprime_month_last"])
    t = None
    for d in dfs:
        cols = [str(x).lower() for x in d.columns]
        if any("дата" in x for x in cols) and any(("знач" in x) or ("став" in x) for x in cols):
            t = d; break
    if t is None: t = dfs[0]
    t.columns = [str(c).strip() for c in t.columns]
    date_col = next((c for c in t.columns if "ата" in c or "date" in c.lower()), t.columns[0])
    val_col  = next((c for c in t.columns if "знач" in c.lower() or "став" in c.lower()), t.columns[-1])

    t = t[[date_col, val_col]].rename(columns={date_col:"date", val_col:"mosprime"})
    t["date"] = pd.to_datetime(t["date"], dayfirst=True, errors="coerce")
    t["mosprime"] = pd.to_numeric(t["mosprime"].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    t.dropna(subset=["date","mosprime"], inplace=True)
    t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
    out = (t.sort_values("date").groupby("month", as_index=False).agg(mosprime_month_last=("mosprime","last")))
    out["mosprime_month_last"] = out["mosprime_month_last"].round(4)
    cache_set(key, out)
    return out

def fetch_imoex_month_close(from_date=None, to_date=None, verify=True) -> pd.DataFrame:
    # Проще без boards: работает стабильно
    base = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    params = {}
    if from_date: params["from"] = pd.to_datetime(from_date).strftime("%Y-%m-%d")
    if to_date:   params["till"] = pd.to_datetime(to_date).strftime("%Y-%m-%d")
    key = f"IMOEX::{json.dumps(params, sort_keys=True)}::{int(verify)}"
    c = cache_get(key)
    if c:
        df = pd.DataFrame(c); df["month"] = pd.to_datetime(df["month"]); return df

    all_rows, start = [], 0
    while True:
        q = dict(params); q["start"] = start
        r = requests.get(base, params=q, timeout=30, verify=verify); r.raise_for_status()
        js = r.json()
        cols = js.get("history", {}).get("columns", [])
        data = js.get("history", {}).get("data", [])
        if not data: break
        df = pd.DataFrame(data, columns=cols)
        if not df.empty:
            df = df[["TRADEDATE","CLOSE"]].copy()
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
            df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
            df.dropna(subset=["TRADEDATE","CLOSE"], inplace=True)
            all_rows.append(df)
        ccols = js.get("history.cursor", {}).get("columns", [])
        cdata = js.get("history.cursor", {}).get("data", [])
        if not cdata: break
        curs = dict(zip(ccols, cdata[0]))
        total = curs.get("TOTAL", 0); rto = curs.get("RANGE_TO", -1)
        start = rto + 1
        if start >= total: break

    if not all_rows:
        log.warning("IMOEX: не получили данных")
        return pd.DataFrame(columns=["month","imoex_month_close"])

    h = pd.concat(all_rows, ignore_index=True)
    h["month"] = h["TRADEDATE"].dt.to_period("M").dt.to_timestamp()
    out = (h.sort_values(["month","TRADEDATE"])
             .groupby("month", as_index=False)
             .agg(imoex_month_close=("CLOSE","last")))
    cache_set(key, out)
    return out

# ----------------------------- #
# ЦБ: M2 и РЕЗЕРВЫ (HTML)
# ----------------------------- #
def fetch_m2_month(verify=True) -> pd.DataFrame:
    """
    Пытаемся скачать M2 из XLSX ЦБ (несколько кандидатов URL).
    Берём первый лист, ищем столбец с 'М2' (или 'M2'), конвертим к month/value.
    """
    candidates = [
        # агрегаты денежной массы (часто обновляется)
        "https://www.cbr.ru/vfs/statistics/credit_statistics/monetary_aggregates.xlsx",
        "https://www.cbr.ru/vfs/statistics/credit_statistics/Monetary_aggregates.xlsx",
        # альтернативный букварь по агрегатам
        "https://www.cbr.ru/vfs/statistics/monetary/monetary_aggregates.xlsx",
    ]
    for url in candidates:
        try:
            log.info(f"GET {url}")
            r = requests.get(url, timeout=40, verify=verify)
            r.raise_for_status()
            # читаем все листы — иногда нужная таблица не на первом
            xls = pd.ExcelFile(BytesIO(r.content))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                if df is None or df.empty:
                    continue
                # нормализуем имена
                df.columns = [str(c).strip() for c in df.columns]
                # ищем колонку даты
                date_col = None
                for c in df.columns:
                    lc = c.lower()
                    if any(k in lc for k in ["дата", "period", "месяц", "month", "год", "year"]):
                        date_col = c
                        break
                if date_col is None:
                    # иногда даты индексом — попробуем по первой колонке
                    date_col = df.columns[0]
                # ищем колонку M2
                m2_col = next((c for c in df.columns if ("м2" in c.lower()) or (c.upper() == "M2")), None)
                if m2_col is None:
                    continue

                tmp = df[[date_col, m2_col]].copy()
                # пробуем распарсить как дату/период
                tmp["month"] = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True)
                # fallback: если это строка вида '2024-01' или 'январь 2024'
                if tmp["month"].isna().all():
                    tmp["month"] = pd.to_datetime(tmp[date_col].astype(str).str.replace(r"[^\d\.\- /]", "", regex=True),
                                                  errors="coerce", dayfirst=True)
                tmp["month"] = tmp["month"].dt.to_period("M").dt.to_timestamp()
                tmp["m2"] = pd.to_numeric(
                    tmp[m2_col].astype(str).str.replace("\u2009", "").str.replace(" ", "").str.replace(",", "."),
                    errors="coerce"
                )
                out = tmp[["month", "m2"]].dropna(subset=["month"]).sort_values("month")
                if not out.empty and out["m2"].notna().any():
                    log.info("Добавлено: m2 (xlsx)")
                    return out
        except Exception as e:
            log.warning(f"M2 XLSX fail {url}: {e}")
    log.warning("M2: не удалось скачать XLSX с ЦБ")
    return pd.DataFrame(columns=["month","m2"])

def fetch_reserves_month(verify=True) -> pd.DataFrame:
    # Международные резервы (ежемесячно)
    url = "https://www.cbr.ru/hd_base/mrrf/mrrf_m/"
    dfs = get_html_tables(url, verify=verify)
    if not dfs:
        log.warning("Reserves: источник недоступен")
        return pd.DataFrame(columns=["month","cb_reserves"])
    # Обычно первая таблица: Дата / млрд долл.
    t = dfs[0].copy()
    t.columns = [str(c).strip() for c in t.columns]
    # ищем «Дата», «Международные резервы» и т.п.
    date_col = next((c for c in t.columns if "ата" in c or "date" in c.lower()), t.columns[0])
    val_col = next((c for c in t.columns if ("резерв" in c.lower()) or ("млрд" in c.lower()) or ("знач" in c.lower())), t.columns[-1])

    t = t[[date_col, val_col]].rename(columns={date_col:"date", val_col:"res"})
    t["date"] = pd.to_datetime(t["date"], dayfirst=True, errors="coerce")
    t["cb_reserves"] = pd.to_numeric(t["res"].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    t.dropna(subset=["date","cb_reserves"], inplace=True)
    t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
    out = (t.sort_values("date").groupby("month", as_index=False).agg(cb_reserves=("cb_reserves","last")))
    log.info("Добавлено: cb_reserves")
    return out

# ----------------------------- #
# РОССТАТ (HTML со страниц; с --insecure при SSL проблемах)
# ----------------------------- #
def fetch_trade_rosstat(verify=True) -> pd.DataFrame:
    url = "https://rosstat.gov.ru/statistics/extern_trade"
    dfs = get_html_tables(url, verify=verify)
    if not dfs:
        log.warning("Trade: не удалось получить таблицы Росстата")
        return pd.DataFrame(columns=["month","trade_export","trade_import"])
    # Ищем таблицу, где есть «экспорт» и «импорт»
    for d in dfs:
        lc = [str(c).lower() for c in d.columns]
        if any("экспорт" in c for c in lc) and any("импорт" in c for c in lc):
            t = d.copy()
            # дата — первая колонка с датами
            mcol = t.columns[0]
            t[mcol] = pd.to_datetime(t[mcol], dayfirst=True, errors="coerce")
            t["month"] = t[mcol].dt.to_period("M").dt.to_timestamp()
            exp_col = next((c for c in t.columns if "экспорт" in str(c).lower()), None)
            imp_col = next((c for c in t.columns if "импорт" in str(c).lower()), None)
            if not exp_col or not imp_col: continue
            t["trade_export"] = pd.to_numeric(str_to_num(t[exp_col]), errors="coerce")
            t["trade_import"] = pd.to_numeric(str_to_num(t[imp_col]), errors="coerce")
            out = t[["month","trade_export","trade_import"]].dropna(subset=["month"])
            if not out.empty:
                log.info("Добавлено: trade_export, trade_import")
                return out
    log.warning("Trade: подходящая таблица не найдена")
    return pd.DataFrame(columns=["month","trade_export","trade_import"])

def fetch_unemp_rosstat(verify=True) -> pd.DataFrame:
    url = "https://rosstat.gov.ru/labor_market_employment_salaries"
    dfs = get_html_tables(url, verify=verify)
    if not dfs:
        log.warning("Unemployment: не получили таблицы")
        return pd.DataFrame(columns=["month","unemp_rate"])
    for d in dfs:
        t = d.copy()
        # пытаемся угадать: колонка с датой + колонка с «уровень безработицы»
        lower_cols = [str(c).lower() for c in t.columns]
        if any("безработиц" in c for c in lower_cols):
            mcol = t.columns[0]
            t[mcol] = pd.to_datetime(t[mcol], dayfirst=True, errors="coerce")
            t["month"] = t[mcol].dt.to_period("M").dt.to_timestamp()
            ucol = next((c for c in t.columns if "безработиц" in str(c).lower()), None)
            if ucol:
                t["unemp_rate"] = pd.to_numeric(str_to_num(t[ucol]), errors="coerce")
                out = t[["month","unemp_rate"]].dropna(subset=["month"])
                if not out.empty:
                    log.info("Добавлено: unemp_rate")
                    return out
    log.warning("Unemployment: подходящая таблица не найдена")
    return pd.DataFrame(columns=["month","unemp_rate"])

def fetch_income_rosstat(verify=True) -> pd.DataFrame:
    url = "https://rosstat.gov.ru/standards_of_living"
    dfs = get_html_tables(url, verify=verify)
    if not dfs:
        log.warning("RealIncome: не получили таблицы")
        return pd.DataFrame(columns=["month","real_income_index"])
    for d in dfs:
        t = d.copy()
        lower_cols = [str(c).lower() for c in t.columns]
        if any(("реальные доходы" in c) or ("индекс реальных" in c) for c in lower_cols):
            mcol = t.columns[0]
            t[mcol] = pd.to_datetime(t[mcol], dayfirst=True, errors="coerce")
            t["month"] = t[mcol].dt.to_period("M").dt.to_timestamp()
            vcol = next((c for c in t.columns if ("реальные доходы" in str(c).lower()) or ("индекс реальных" in str(c).lower())), None)
            if vcol:
                t["real_income_index"] = pd.to_numeric(str_to_num(t[vcol]), errors="coerce")
                out = t[["month","real_income_index"]].dropna(subset=["month"])
                if not out.empty:
                    log.info("Добавлено: real_income_index")
                    return out
    log.warning("RealIncome: подходящая таблица не найдена")
    return pd.DataFrame(columns=["month","real_income_index"])

def fetch_ipi_rosstat(verify=True) -> pd.DataFrame:
    url = "https://rosstat.gov.ru/industrial_production"
    dfs = get_html_tables(url, verify=verify)
    if not dfs:
        log.warning("IPI: не получили таблицы")
        return pd.DataFrame(columns=["month","ipi_index"])
    for d in dfs:
        t = d.copy()
        lower_cols = [str(c).lower() for c in t.columns]
        if any(("индекс промышленного" in c) or ("ипп" in c) for c in lower_cols):
            mcol = t.columns[0]
            t[mcol] = pd.to_datetime(t[mcol], dayfirst=True, errors="coerce")
            t["month"] = t[mcol].dt.to_period("M").dt.to_timestamp()
            vcol = next((c for c in t.columns if ("индекс промышленного" in str(c).lower()) or ("ипп" in str(c).lower())), None)
            if vcol:
                t["ipi_index"] = pd.to_numeric(str_to_num(t[vcol]), errors="coerce")
                out = t[["month","ipi_index"]].dropna(subset=["month"])
                if not out.empty:
                    log.info("Добавлено: ipi_index")
                    return out
    log.warning("IPI: подходящая таблица не найдена")
    return pd.DataFrame(columns=["month","ipi_index"])

def fetch_pmi_rosstat(verify=True) -> pd.DataFrame:
    """
    Пытаемся найти общую таблицу по PMI (производство/услуги). Если не нашли — вернём пусто.
    """
    url = "https://rosstat.gov.ru/business_survey"  # где часто бывают индексы/PMI
    dfs = get_html_tables(url, verify=verify)
    res = []
    if dfs:
        for d in dfs:
            t = d.copy()
            lc = [str(c).lower() for c in t.columns]
            has_pmi = any("pmi" in c for c in lc) or any("деловой активност" in c for c in lc)
            if not has_pmi: continue
            mcol = t.columns[0]
            t[mcol] = pd.to_datetime(t[mcol], dayfirst=True, errors="coerce")
            t["month"] = t[mcol].dt.to_period("M").dt.to_timestamp()
            for c in t.columns:
                cl = str(c).lower()
                if "manufactur" in cl or "производств" in cl:
                    tmp = t[["month", c]].copy()
                    tmp["pmi_manuf"] = pd.to_numeric(str_to_num(tmp[c]), errors="coerce")
                    res.append(tmp[["month","pmi_manuf"]])
                if "services" in cl or "услуг" in cl:
                    tmp = t[["month", c]].copy()
                    tmp["pmi_services"] = pd.to_numeric(str_to_num(tmp[c]), errors="coerce")
                    res.append(tmp[["month","pmi_services"]])
    if not res:
        log.warning("PMI: не нашли таблицу на Росстате; заполняем NaN")
        return pd.DataFrame(columns=["month","pmi_manuf","pmi_services"])
    # Свести всё, что нашли
    out = None
    for part in res:
        out = part if out is None else out.merge(part, on="month", how="outer")
    out = out.sort_values("month")
    # возможно много NaN — это нормально
    return out

def str_to_num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", ".").str.replace(" ", "").str.replace("\u2009","")

# ----------------------------- #
# MAIN
# ----------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Обогащение новостей макро- и финпоказателями (сеть, без локальных CSV).")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--dayfirst", action="store_true")
    ap.add_argument("--allow-network", action="store_true", help="включает сетевые источники (ЦБ/МОЕХ/Росстат)")
    ap.add_argument("--insecure", action="store_true", help="verify=False для источников с проблемными сертификатами (Росстат)")
    ap.add_argument("--fx", nargs="*", default=["USD","EUR","CNY"], help="валюты для месячного среднего из ЦБ")
    args = ap.parse_args()

    verify = not args.insecure

    df = pd.read_csv(args.in_csv)
    if args.date_col not in df.columns:
        log.error(f"Нет колонки '{args.date_col}'. Есть: {list(df.columns)}"); sys.exit(1)

    df["month"] = df[args.date_col].apply(lambda x: to_month(x, dayfirst=args.dayfirst))
    if df["month"].isna().any():
        log.warning("Некоторые даты не распознаны — для них признаки будут NaN.")
    mn, mx = month_span(df["month"])

    out = df.copy()

    if not args.allow_network:
        log.warning("Флаг --allow-network не указан: ничего из сети не будет скачиваться.")
        out.to_csv(args.out_csv, index=False, encoding="utf-8")
        log.info(f"Готово: {os.path.abspath(args.out_csv)}")
        return

    # ---- ЦБ, МОЕХ (финансы) ----
    try:
        ruonia = fetch_ruonia_month_avg(verify=verify)
        out = out.merge(ruonia, on="month", how="left")
        log.info("Добавлено: ruonia_avg_month_pct")
    except Exception as e:
        log.warning(f"RUONIA: {e}")

    try:
        mos = fetch_mosprime_month_last(verify=verify)
        out = out.merge(mos, on="month", how="left")
        log.info("Добавлено: mosprime_month_last")
    except Exception as e:
        log.warning(f"MosPrime: {e}")

    try:
        if mn is not None and mx is not None:
            imoex = fetch_imoex_month_close(from_date=mn, to_date=mx, verify=verify)
        else:
            imoex = fetch_imoex_month_close(verify=verify)
        out = out.merge(imoex, on="month", how="left")
        log.info("Добавлено: imoex_month_close")
    except Exception as e:
        log.warning(f"IMOEX: {e}")

    # ---- ЦБ: M2, резервы ----
    try:
        m2 = fetch_m2_month(verify=verify)
        if not m2.empty:
            out = out.merge(m2, on="month", how="left")
    except Exception as e:
        log.warning(f"M2: {e}")

    try:
        resv = fetch_reserves_month(verify=verify)
        if not resv.empty:
            out = out.merge(resv, on="month", how="left")
    except Exception as e:
        log.warning(f"Reserves: {e}")

    # ---- Росстат: торговля, безработица, доходы, ИПП, PMI ----
    try:
        trade = fetch_trade_rosstat(verify=verify)
        if not trade.empty:
            out = out.merge(trade, on="month", how="left")
    except Exception as e:
        log.warning(f"Trade: {e}")

    try:
        unemp = fetch_unemp_rosstat(verify=verify)
        if not unemp.empty:
            out = out.merge(unemp, on="month", how="left")
    except Exception as e:
        log.warning(f"Unemployment: {e}")

    try:
        rincome = fetch_income_rosstat(verify=verify)
        if not rincome.empty:
            out = out.merge(rincome, on="month", how="left")
    except Exception as e:
        log.warning(f"RealIncome: {e}")

    try:
        ipi = fetch_ipi_rosstat(verify=verify)
        if not ipi.empty:
            out = out.merge(ipi, on="month", how="left")
    except Exception as e:
        log.warning(f"IPI: {e}")

    try:
        pmi = fetch_pmi_rosstat(verify=verify)
        if not pmi.empty:
            out = out.merge(pmi, on="month", how="left")
            # если вдруг несколько значений в один месяц — оставим последнее
            out = (out.sort_values("month")
                      .groupby(list(out.columns), axis=0, as_index=False).first())
    except Exception as e:
        log.warning(f"PMI: {e}")

    # ---- FX среднее за месяц (ЦБ XML_daily) ----
    try:
        months = out["month"].dropna().unique()
        fx = fx_monthly_avg(months, codes=tuple(args.fx), verify=verify)
        out = out.merge(fx, on="month", how="left")
        log.info(f"Добавлены FX: {', '.join(c+'_avg_month_cbr' for c in args.fx)}")
    except Exception as e:
        log.warning(f"FX: {e}")

    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    log.info(f"Готово: {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()