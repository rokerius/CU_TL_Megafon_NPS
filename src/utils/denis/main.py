import pandas as pd
from src.utils.denis.economic_data.parser import create_household_params_fast, create_potreb_prices_params, create_key_rate_params
from src.utils.denis.weather_data.main import create_new_weather_columns
from src.utils.denis.economic_data.reader import df_households, potreb_prices, key_rate_data


def prepare_data(df_input, weather_cache_path='cache/weather_cache.pkl'):
    df_input = create_household_params_fast(df_input, df_households)
    df_input = create_potreb_prices_params(df_input, potreb_prices)
    df_input = create_key_rate_params(df_input, key_rate_data)
    print("economic data added!")
    df_result = create_new_weather_columns(df_input, cache_path=weather_cache_path)
    print("weather data added!")
    
    # СЮДА ВСТАВЛЯТЬ ОБРАБОТКУ ДАННЫХ!!!
    
    return df_result

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