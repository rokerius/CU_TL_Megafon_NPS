import pandas as pd
from src.utils.denis.economic_data.parser import create_household_params_fast, create_potreb_prices_params, create_key_rate_params, create_exchange_features
from src.utils.denis.weather_data.main import create_new_weather_columns
from src.utils.denis.economic_data.reader import df_households, potreb_prices, key_rate_data, exchange_rates_data

data_dict = {
    "households": df_households,
    "potreb_prices": potreb_prices,
    "key_rate_data": key_rate_data,
    "exchange_rates_data": exchange_rates_data
}


def prepare_data(df_input, weather_cache_path='cache/weather_cache.pkl', data_dict=data_dict):
    df_input = create_household_params_fast(df_input, data_dict["households"])
    df_input = create_potreb_prices_params(df_input, data_dict["potreb_prices"])
    df_input = create_key_rate_params(df_input, data_dict["key_rate_data"])
    df_input = create_exchange_features(df_input, data_dict["exchange_rates_data"])
    print("economic data added!")
    df_result = create_new_weather_columns(df_input, cache_path=weather_cache_path)
    print("weather data added!")
    
    
    # СЮДА ВСТАВЛЯТЬ ОБРАБОТКУ ДАННЫХ!!!
    
    
    null_cols = df_result.columns[df_result.isnull().any()]
    print("Столбцы с пропусками:", null_cols.tolist())

    df_result = df_result.drop(null_cols.tolist(), axis=1)
    print("Оставшиеся столбцы:", df_result.columns.tolist())
    
    df_result_encoded = pd.get_dummies(df_result, columns=['state'])
    
    return df_result_encoded

def main():
    df_input = pd.DataFrame({
        "year": [2020, 2018, 2025],
        "month": [2, 6, 1],
        "state": ["Алтайский край", "Алтайский край", "Алтайский край"]
    })

    df_result = prepare_data(df_input)
    print(df_result.head(5))
    print(df_result.columns)


if __name__ == "__main__":
    main()