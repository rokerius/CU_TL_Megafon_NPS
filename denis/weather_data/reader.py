import pandas as pd

df = pd.read_excel("denis/weather_data/target.xlsx", sheet_name="data cleaned")
states = df["state"].unique()