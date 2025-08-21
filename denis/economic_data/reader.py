import pandas as pd

# Загружаем Excel
df = pd.read_excel("data/households_b_3.xlsx", "Балансы")
# начиная с 6-й строки (нумерация с 0)

# Убираем пустые колонки
df = df.dropna(axis=1, how="all")

# Преобразуем таблицу в формат long
df = df.rename(columns={"Дата": "Параметр"})

# melt: разворачиваем даты в строки
df_long = df.melt(
    id_vars=["Параметр"],       # это строки с названиями показателей
    var_name="Дата",            # сюда попадут даты
    value_name="Значение"       # сюда попадут значения
)

# приводим даты к datetime
df_long["Дата"] = pd.to_datetime(df_long["Дата"], errors="coerce")

# убираем NaT (если вдруг затесались строки не с датами)
df_long = df_long.dropna(subset=["Дата"]).reset_index(drop=True)


def main():
    print(df.head(20))
    print(df_long.head(20))
    print(df_long.info())
    
if __name__ == "__main__":
    main()
    
