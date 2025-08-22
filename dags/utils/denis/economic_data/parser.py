import pandas as pd
from utils.denis.economic_data.reader import df_long


def create_household_params(df_input):
    df_input["target_date"] = pd.to_datetime(
        df_input.assign(day=1)[["year", "month", "day"]]
    )

    unique_params = df_long["Параметр"].unique()
    all_pairs = pd.MultiIndex.from_product(
        [df_input.index, unique_params], names=["row_id", "Параметр"]
    ).to_frame(index=False)

    all_pairs = all_pairs.merge(
        df_input[["target_date"]], left_on="row_id", right_index=True
    )

    merged = all_pairs.merge(df_long, on="Параметр", how="left")

    merged["diff"] = (merged["Дата"] - merged["target_date"]).abs()

    closest = merged.loc[merged.groupby(["row_id", "Параметр"])["diff"].idxmin()]

    pivot = closest.pivot(index="row_id", columns="Параметр", values="Значение")

    # Шаг 7. Приклеиваем обратно к df_input
    df_result = pd.concat([df_input, pivot], axis=1)
    df_result.drop("target_date", axis=1, inplace=True)
    
    return df_result

def main():
    df_input = pd.DataFrame({
        "year": [2020, 2018, 2025],
        "month": [2, 6, 1]
    })

    df_result = create_household_params(df_input)
    print(df_result.head(5))
    
if __name__ == "__main__":
    main()
