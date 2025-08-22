import pandas as pd

df = pd.read_excel("data/households_b_3.xlsx", "Балансы")
df = df.dropna(axis=1, how="all")
df = df.rename(columns={"Дата": "Параметр"})

df_long = df.melt(
    id_vars=["Параметр"],
    var_name="Дата",
    value_name="Значение")

df_long["Дата"] = pd.to_datetime(df_long["Дата"], errors="coerce")
df_long = df_long.dropna(subset=["Дата"]).reset_index(drop=True)


def main():
    print(df.head(20))
    print(df_long.head(20))
    print(df_long.info())
    
if __name__ == "__main__":
    main()
    
