import pandas as pd
import numpy as np
from reader import df_households
from parser import create_household_params


df_input = pd.DataFrame({
    "year": [2020, 2018, 2025],
    "month": [2, 6, 1]
})

df_result = create_household_params(df_input, df_households)


def main():
    print(df_result.head(5))
    print(df_result.info())
    
if __name__ == "__main__":
    main()
