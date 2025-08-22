import pandas as pd

df = pd.read_excel("data/target.xlsx", sheet_name="data cleaned")
states = df["state"].unique()
