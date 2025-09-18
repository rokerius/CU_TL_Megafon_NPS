#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from pandas.api.types import is_numeric_dtype


# ------------------ утилиты ------------------
def sanitize_name(name: str) -> str:
    name = str(name).strip().replace("/", "÷").replace("\\", "÷")
    name = re.sub(r'[\n\r\t]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name[:120]


def safe_partial_dependence(gam, term, XX, want_ci=True):
    """ Возвращает (pdp_raw, ci_raw|None) для разных версий pygam. """
    try:
        res = gam.partial_dependence(term=term, X=XX, width=0.95) if want_ci \
              else gam.partial_dependence(term=term, X=XX)
    except TypeError:
        res = gam.partial_dependence(term=term, X=XX)

    pdp, ci = None, None
    if isinstance(res, tuple):
        if len(res) >= 1:
            pdp = res[0]
        if len(res) >= 2:
            ci = res[1]
    else:
        pdp = res
    return pdp, ci


def _to_1d_center_from_two(curve_a, curve_b):
    a = np.asarray(curve_a).ravel()
    b = np.asarray(curve_b).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    center = (a + b) / 2.0
    return center, a, b


def normalize_pdp_output(pdp, ci, grid_len):
    """
    Приводит выход PDP к 1D. Возвращает (pdp_1d, ci_lower, ci_upper),
    где ci_* могут быть None. Терпит:
      - 1D массивы
      - 2D массивы вида (2, N) / (N, 2), в том числе dtype=object с вложенными массивами
      - списки длины 2 [lower, upper]
      - экзотические формы — берём среднее по нужной оси / обрезаем.
    """
    # 1) если pdp — список длины 2 (ниж/верх), делаем центр
    if isinstance(pdp, (list, tuple)) and len(pdp) == 2:
        center, lo, hi = _to_1d_center_from_two(pdp[0], pdp[1])
        return center[:grid_len], lo[:grid_len], hi[:grid_len]

    # 2) объектные массивы (например, shape (2, N) dtype=object)
    try:
        pdp_arr = np.asarray(pdp)
    except Exception:
        # последний шанс: сделаем 1D из первого элемента
        pdp_arr = np.asarray(pdp[0])

    # 1D — идеально
    if pdp_arr.ndim == 1:
        pdp_1d = pdp_arr.ravel()[:grid_len]
        ci_lo = ci_hi = None
        if ci is not None:
            try:
                ci_arr = np.asarray(ci)
                if ci_arr.ndim == 2 and ci_arr.shape[0] == 2:
                    ci_lo, ci_hi = ci_arr[0].ravel()[:grid_len], ci_arr[1].ravel()[:grid_len]
                elif ci_arr.ndim == 2 and ci_arr.shape[1] == 2:
                    ci_lo, ci_hi = ci_arr[:, 0].ravel()[:grid_len], ci_arr[:, 1].ravel()[:grid_len]
            except Exception:
                pass
        return pdp_1d, ci_lo, ci_hi

    # 2D — возможные варианты
    if pdp_arr.ndim == 2:
        r, c = pdp_arr.shape
        # объектный (2, N): элементы — массивы
        if r == 2 and pdp_arr.dtype == object:
            return _to_1d_center_from_two(pdp_arr[0], pdp_arr[1])
        if c == 2 and pdp_arr.dtype == object:
            return _to_1d_center_from_two(pdp_arr[:, 0], pdp_arr[:, 1])

        # обычный числовой (2, N) -> средняя линия, CI как полосы
        if r == 2 and c >= grid_len:
            lo = np.asarray(pdp_arr[0]).ravel()[:grid_len]
            hi = np.asarray(pdp_arr[1]).ravel()[:grid_len]
            center = (lo + hi) / 2.0
            return center, lo, hi
        if c == 2 and r >= grid_len:
            lo = np.asarray(pdp_arr[:, 0]).ravel()[:grid_len]
            hi = np.asarray(pdp_arr[:, 1]).ravel()[:grid_len]
            center = (lo + hi) / 2.0
            return center, lo, hi

        # экзотика: усредним по оси (подгоняем к grid_len)
        if r == grid_len:
            center = np.nanmean(pdp_arr, axis=1).ravel()[:grid_len]
        elif c == grid_len:
            center = np.nanmean(pdp_arr, axis=0).ravel()[:grid_len]
        else:
            center = pdp_arr.ravel()[:grid_len]
        return center, None, None

    # >2D — обрежем / расплющим
    pdp_flat = pdp_arr.ravel()[:grid_len]
    return pdp_flat, None, None


def safe_spearman(x, y):
    c, p = spearmanr(x, y, nan_policy="omit")
    if np.isnan(c): c = 0.0
    if np.isnan(p): p = 1.0
    return float(c), float(p)


# ------------------ основной скрипт ------------------
def main():
    ap = argparse.ArgumentParser(
        description="Интерпретируемая модель (pyGAM) + PDP + эластичность + важность"
    )
    ap.add_argument("--csv", default="vlad/train_final.csv", help="Путь к CSV с данными")
    ap.add_argument("--target-col", default=None,
                    help="Имя таргета (если не задано — пытаемся nps, потом target)")
    ap.add_argument("--date-cols", nargs="*", default=["start_date", "date", "month"],
                    help="Колонки с датой (исключаются из X)")
    ap.add_argument("--max-gam-features", type=int, default=50,
                    help="Сколько фич взять в GAM после отбора по Spearman")
    ap.add_argument("--out-dir", default="gam_reports", help="Куда класть графики и отчёт")
    ap.add_argument("--test-size", type=float, default=0.2, help="Размер валидационной выборки")
    ap.add_argument("--random-state", type=int, default=42, help="seed")
    ap.add_argument("--plots-limit", type=int, default=7, help="Максимум итоговых графиков")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Загружаю:", args.csv)
    df = pd.read_csv(args.csv)

    # приведение дат
    for c in args.date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # выбор таргета
    target_col = args.target_col or ("nps" if "nps" in df.columns else ("target" if "target" in df.columns else None))
    if not target_col or target_col not in df.columns:
        raise ValueError("Не нашёл колонку таргета. Укажи её через --target-col")

    y = pd.to_numeric(df[target_col], errors="coerce")
    idx = y.notna()
    df, y = df.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

    # кандидаты фич — числовые, исключаем даты и таргет, убираем константы
    num_cols = []
    for c in df.columns:
        if c in args.date_cols or c == target_col:
            continue
        if is_numeric_dtype(df[c]) and df[c].dropna().nunique() > 1:
            num_cols.append(c)
    if not num_cols:
        raise ValueError("После фильтров не осталось числовых признаков.")

    # отбор по Spearman
    corr_rows = []
    for c in num_cols:
        corr, p = safe_spearman(pd.to_numeric(df[c], errors="coerce"), y)
        corr_rows.append((c, abs(corr), corr, p))
    corr_df = pd.DataFrame(corr_rows, columns=["feature", "abs_spearman", "spearman", "spearman_p"])
    corr_df.sort_values("abs_spearman", ascending=False, inplace=True)

    gam_features = corr_df.head(min(args.max_gam_features, len(corr_df)))["feature"].tolist()

    # матрица X
    X = df[gam_features].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=gam_features, index=X.index)

    # трейн/валид
    Xtr, Xva, ytr, yva = train_test_split(
        X_imp.values, y.values, test_size=args.test_size, random_state=args.random_state
    )

    # термы сплайнов
    terms = s(0)
    for i in range(1, X_imp.shape[1]):
        terms = terms + s(i)

    print("[INFO] Обучаем GAM…")
    lam_grid = np.logspace(-2, 2, 6)  # 0.01 .. 100
    gam = LinearGAM(terms).gridsearch(Xtr, ytr, lam=lam_grid)

    r2_tr = gam.score(Xtr, ytr)
    r2_va = gam.score(Xva, yva)
    print(f"[INFO] R^2 train: {r2_tr:.4f}")
    print(f"[INFO] R^2 valid: {r2_va:.4f}")

    # ===== Важность (пермутация) =====
    print("[INFO] Считаю пермутационную важность…")
    pi = permutation_importance(gam, Xva, yva, n_repeats=30, random_state=args.random_state)
    perm_imp = pd.DataFrame({
        "feature": gam_features,
        "perm_importance_mean": pi.importances_mean,
        "perm_importance_std":  pi.importances_std
    })

    # ===== PDP и эластичности (+ summary) =====
    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []
    for i, col in enumerate(gam_features):
        XX = gam.generate_X_grid(term=i)
        grid = XX[:, i]

        raw_pdp, raw_ci = safe_partial_dependence(gam, term=i, XX=XX, want_ci=True)
        pdp, ci_lo, ci_hi = normalize_pdp_output(raw_pdp, raw_ci, grid_len=len(grid))

        # численная производная как эластичность
        elastic = np.gradient(pdp, grid)

        summary_rows.append({
            "feature": col,
            "avg_abs_elasticity": float(np.nanmean(np.abs(elastic))),
            "median_elasticity": float(np.nanmedian(elastic)),
        })

    elastic_df = pd.DataFrame(summary_rows)

    # общий summary
    summary = (corr_df.merge(elastic_df, on="feature", how="inner")
                        .merge(perm_imp, on="feature", how="left"))
    # нормировки и сводный балл
    for col in ["abs_spearman", "avg_abs_elasticity", "perm_importance_mean"]:
        m = np.nanmax(np.abs(summary[col].values)) if len(summary) else 0.0
        summary[col + "_norm"] = summary[col] / m if m and m > 0 else 0.0
    summary["driver_score"] = (
        0.4 * summary["avg_abs_elasticity_norm"] +
        0.4 * summary["perm_importance_mean_norm"] +
        0.2 * summary["abs_spearman_norm"]
    )
    summary.sort_values("driver_score", ascending=False, inplace=True)
    summary_path = os.path.join(args.out_dir, "drivers_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # ===== Графики (до args.plots_limit) =====
    def save_barh(df_plot, value_col, title, filename, xlab):
        plt.figure(figsize=(9, 6))
        plt.barh(df_plot["feature"][::-1], df_plot[value_col][::-1])
        plt.title(title)
        plt.xlabel(xlab)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, filename), dpi=150)
        plt.close()

    # 1: Топ по |эластичности|
    top_el = summary.nlargest(min(15, args.plots_limit*3), "avg_abs_elasticity").copy()
    save_barh(top_el, "avg_abs_elasticity",
              "Топ по средней |эластичности|", "01_top_avg_abs_elasticity.png",
              "avg |dPDP/dx|")

    # 2: Топ по пермутационной важности
    top_pi = summary.nlargest(min(15, args.plots_limit*3), "perm_importance_mean").copy()
    plt.figure(figsize=(9, 6))
    plt.barh(top_pi["feature"][::-1], top_pi["perm_importance_mean"][::-1],
             xerr=top_pi["perm_importance_std"][::-1])
    plt.title("Топ по пермутационной важности")
    plt.xlabel("Permutation importance (valid)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "02_top_permutation_importance.png"), dpi=150)
    plt.close()

    # 3: Топ по |Spearman|
    top_sp = summary.nlargest(min(15, args.plots_limit*3), "abs_spearman").copy()
    save_barh(top_sp, "abs_spearman",
              "Топ по |Spearman| с NPS", "03_top_abs_spearman.png",
              "|Spearman|")

    # 4: Корр-матрица ТОП-20 по driver_score
    topN = min(20, max(10, args.plots_limit*3))
    top_feats = summary.head(topN)["feature"].tolist()
    corrm = pd.DataFrame(X_imp, columns=gam_features)[top_feats].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corrm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Spearman-корреляции между ТОП драйверами")
    plt.xticks(range(len(top_feats)), [sanitize_name(c) for c in top_feats], rotation=90)
    plt.yticks(range(len(top_feats)), [sanitize_name(c) for c in top_feats])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "04_corr_heatmap_top.png"), dpi=150)
    plt.close()

    # 5: PDP-грид (ТОП-4 по driver_score)
    top4 = summary.head(4)["feature"].tolist()
    plt.figure(figsize=(12, 8))
    for j, col in enumerate(top4, 1):
        i = gam_features.index(col)
        XX = gam.generate_X_grid(term=i)
        grid = XX[:, i]
        raw_pdp, raw_ci = safe_partial_dependence(gam, term=i, XX=XX, want_ci=True)
        pdp, ci_lo, ci_hi = normalize_pdp_output(raw_pdp, raw_ci, grid_len=len(grid))

        ax = plt.subplot(2, 2, j)
        ax.plot(grid, pdp, lw=2)
        if ci_lo is not None and ci_hi is not None:
            ax.fill_between(grid, ci_lo, ci_hi, alpha=0.2)
        ax.set_title(f"PDP: {col}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "05_pdp_grid_top4.png"), dpi=150)
    plt.close()

    # 6: Эластичности для ТОП-4
    plt.figure(figsize=(12, 8))
    for j, col in enumerate(top4, 1):
        i = gam_features.index(col)
        XX = gam.generate_X_grid(term=i)
        grid = XX[:, i]
        raw_pdp, _ = safe_partial_dependence(gam, term=i, XX=XX, want_ci=False)
        pdp, _, _ = normalize_pdp_output(raw_pdp, None, grid_len=len(grid))
        elastic = np.gradient(pdp, grid)

        ax = plt.subplot(2, 2, j)
        ax.plot(grid, elastic, lw=2)
        ax.axhline(0, ls="--", alpha=0.5)
        ax.set_title(f"Эластичность: d({target_col})/d({col})")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "06_elasticity_grid_top4.png"), dpi=150)
    plt.close()

    # 7: Направление влияния — медианная эластичность ТОП-12 по driver_score
    top12 = summary.head(12).copy()
    plt.figure(figsize=(9, 6))
    plt.barh(top12["feature"][::-1], top12["median_elasticity"][::-1])
    plt.title("Направление влияния: медианная эластичность (ТОП по driver_score)")
    plt.xlabel("median dPDP/dx ( >0 — в среднем повышает NPS )")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "07_direction_median_elasticity_top.png"), dpi=150)
    plt.close()

    # Текстовый отчёт
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Итоговый отчёт по драйверам NPS (pyGAM) ===\n\n")
        f.write(f"Файл: {os.path.basename(args.csv)}\n")
        f.write(f"Таргет: {target_col}\n")
        f.write(f"Размер выборки: {len(df)} наблюдений, фич в GAM: {len(gam_features)}\n")
        f.write(f"Качество модели: R^2 train={r2_tr:.4f}, valid={r2_va:.4f}\n")
        f.write(f"Лучшая lam: {gam.lam}\n\n")
        f.write("ТОП-драйверы по сводному баллу driver_score:\n")
        for _, row in summary.head(15).iterrows():
            f.write(f" - {row['feature']}: score={row['driver_score']:.3f}, "
                    f"avg|elastic|={row['avg_abs_elasticity']:.4f}, "
                    f"perm_imp={row.get('perm_importance_mean', np.nan):.4f}"
                    f"±{row.get('perm_importance_std', np.nan):.4f}, "
                    f"spearman={row['spearman']:.3f} (p={row['spearman_p']:.3g})\n")

    print("[INFO] Готово. Графики и отчёты в:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()