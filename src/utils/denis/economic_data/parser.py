import pandas as pd


REGION_ALIASES = {
    "Москва": "г. Москва",
    "Санкт-Петербург": "г.Санкт-Петербург",
    "Республика Северная Осетия - Алания": "Республика Северная Осетия\n - Алания",
    "Ханты-Мансийский автономный округ - Югра": "Ханты-Мансийский авт.округ - Югра",
    "Кемеровская область": "Кемеровская область - Кузбасс",
    'Чувашская Республика - Чувашия': 'Чувашская Республика',
    'Республика Татарстан (Татарстан)': 'Республика Татарстан'
    
}


def create_household_params_fast(df_input: pd.DataFrame, df_households: pd.DataFrame) -> pd.DataFrame:
    df_input = df_input.copy()
    df_input = df_input.reset_index(drop=True)   # гарантируем чистую индексацию
    df_input["row_id"] = df_input.index          # сохраняем ID для связи
    
    df_input["target_date"] = pd.to_datetime(
        df_input.assign(day=1)[["year", "month", "day"]]
    )

    results = []

    for param, subset in df_households.groupby("Параметр"):
        subset_sorted = subset.sort_values("Дата")
        merged = pd.merge_asof(
            df_input.sort_values("target_date"),
            subset_sorted,
            left_on="target_date",
            right_on="Дата",
            direction="nearest"
        )
        merged["Параметр"] = param
        results.append(merged[["row_id", "Параметр", "Значение"]])

    pivot = pd.concat(results).pivot(
        index="row_id", columns="Параметр", values="Значение"
    )

    # Возвращаем в исходный порядок + данные
    df_out = df_input.drop(columns=["target_date"])
    df_out = df_out.merge(pivot, left_on="row_id", right_index=True)
    df_out = df_out.drop(columns=["row_id"])
    return df_out



def get_potreb_info(year, month, region=None, code=None, potreb_prices=None):
    df = potreb_prices.get(f"{month}/{year}")
    if code is not None:
        row = df[df["Код территории"] == code]
    elif region is not None:
        # нормализуем название
        region_norm = REGION_ALIASES.get(region, region)
        if region_norm == "г. Москва" and year == 2025:
            region_norm = "г.Москва"
        row = df[df["Наименование территории"].str.strip() == region_norm]

        # на случай проблем с переносами строк и пробелами
        if row.empty:
            row = df[df["Наименование территории"].str.replace("\s+", " ", regex=True).str.strip() == region_norm]
    else:
        raise ValueError("Нужно указать либо 'region', либо 'code'")

    if row.empty:
        return None
    
    result = row.iloc[0].to_dict()
    result.pop("Код территории", None)
    result.pop("Наименование территории", None)
    return result


def create_potreb_prices_params(df_input, potreb_prices):
    results = df_input.apply(
        lambda row: get_potreb_info(
            year=row["year"],
            month=row["month"],
            region=row.get("state", None),
            code=row.get("code", None),
            potreb_prices=potreb_prices
        ) or {},
        axis=1
    )

    results_df = pd.DataFrame(results.tolist())

    return pd.concat([df_input.reset_index(drop=True), results_df], axis=1)


def create_key_rate_params(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Дата"] = df["month"].apply(lambda x: f"{int(x):02d}") + "." + df["year"].astype(str)
    df["Дата"] = df["Дата"].astype(str)
    ref_df["Дата"] = ref_df["Дата"].astype(str)
    merged = df.merge(ref_df, on="Дата", how="left")
    merged.drop(columns=["Дата"], inplace=True)
    return merged


def main():
    df_input = pd.DataFrame({
        "year": [2020, 2018, 2025],
        "month": [2, 6, 1]
    })

    # Тестовый df_long
    df_long = pd.DataFrame({
        "Параметр": ["A", "A", "A", "B", "B"],
        "Дата": pd.to_datetime(["2019-01-01", "2020-02-01", "2025-01-01",
                                "2018-06-01", "2020-02-15"]),
        "Значение": [10, 20, 30, 100, 200]
    })

    df_result = create_household_params_fast(df_input, df_long)
    print(df_result)


if __name__ == "__main__":
    main()
