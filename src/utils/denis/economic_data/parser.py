import pandas as pd


def create_household_params_fast(df_input: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    df_input = df_input.copy()
    df_input = df_input.reset_index(drop=True)   # гарантируем чистую индексацию
    df_input["row_id"] = df_input.index          # сохраняем ID для связи
    
    df_input["target_date"] = pd.to_datetime(
        df_input.assign(day=1)[["year", "month", "day"]]
    )

    results = []

    for param, subset in df_long.groupby("Параметр"):
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
