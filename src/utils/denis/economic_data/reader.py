import pandas as pd


def prepare_househols_data(df):
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns={"Дата": "Параметр"})

    df_long = df.melt(
        id_vars=["Параметр"],
        var_name="Дата",
        value_name="Значение")

    df_long["Дата"] = pd.to_datetime(df_long["Дата"], errors="coerce")
    df_long = df_long.dropna(subset=["Дата"]).reset_index(drop=True)
    return df_long


def prepare_potreb_prices_data(df):
    df.columns = [
        "Код территории",
        "Наименование территории",
        "ИПЦ",
        "Продовольственные товары",
        "Непродовольственные товары",
        "Услуги"
    ]

    df = df.dropna(subset=["Код территории"])
    for col in ["ИПЦ", "Продовольственные товары", "Непродовольственные товары", "Услуги"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    return df

def create_potreb_prices_dfs(path: str):
    potreb_prices = {}
    for month in range(1, 10):
        df = pd.read_excel(path, sheet_name="0"+str(month)+"(2024)", skiprows=6)
        potreb_prices[str(month)+"/2024"] = prepare_potreb_prices_data(df)
    for month in range(10, 13):
        df = pd.read_excel(path, sheet_name=str(month)+"(2024)", skiprows=6)
        potreb_prices[str(month)+"/2024"] = prepare_potreb_prices_data(df)
        
    df = pd.read_excel(path, sheet_name="01(2025)", skiprows=6)
    potreb_prices["1/2025"] = prepare_potreb_prices_data(df)
    return potreb_prices


    
    
df_households = prepare_househols_data(pd.read_excel("data/households_b_3.xlsx", "Балансы")) 
potreb_prices = create_potreb_prices_dfs("data/potreb_prices.xlsx")
key_rate_data = pd.read_excel("data/key_rate.xlsx")


def main():
    print(df_households.head(20))
    print(df_households.info())
    
if __name__ == "__main__":
    main()
    
