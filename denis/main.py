import pandas as pd
from src.utils.denis.main import prepare_data
from src.utils.denis.weather_data.main import create_new_weather_columns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    df_input = pd.read_excel("data/target.xlsx", sheet_name="data cleaned")

    df = prepare_data(df_input)
    print(df.head(5))
    print(df.columns)
    
    feature_cols = [col for col in df.columns if col not in ["start_date", "year", "month", "nps", "state"]]
    X = df[feature_cols]
    y = df["nps"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=239)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()
