import pandas as pd
from utils.denis.economic_data.parser import create_household_params
from utils.denis.weather_data.main import create_new_weather_columns


def prepare_data(df_input):
    df_result = create_household_params(df_input)
    df_result = create_new_weather_columns(df_result)
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