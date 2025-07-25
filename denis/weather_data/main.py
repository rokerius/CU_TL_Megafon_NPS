import requests
import pickle
import pandas as pd
import calendar
from tqdm import tqdm
import time
import os
import datetime
from config import categories, regions_coords


import requests
import pandas as pd

def get_weather_daily_stats(latitude: float, longitude: float,
                            start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,weathercode,"
        "relative_humidity_2m_max,relative_humidity_2m_mean,cloudcover_mean,"
        "shortwave_radiation_sum,snowfall_sum"
        "&timezone=Europe/Moscow"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise Exception("Timeout error при запросе погоды")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка запроса: {e}")

    data = response.json()
    daily = data.get('daily', {})

    df_daily = pd.DataFrame({
        'date': daily.get('time', []),
        'temp_max': daily.get('temperature_2m_max', []),
        'temp_min': daily.get('temperature_2m_min', []),
        'precipitation': daily.get('precipitation_sum', []),
        'wind_speed_max': daily.get('wind_speed_10m_max', []),
        'weathercode': daily.get('weathercode', []),
        'humidity_max': daily.get('relative_humidity_2m_max', []),
        'humidity_mean': daily.get('relative_humidity_2m_mean', []),
        'cloudcover': daily.get('cloudcover_mean', []),
        'solar_radiation': daily.get('shortwave_radiation_sum', []),
        'snowfall': daily.get('snowfall_sum', [])
    })

    return df_daily



def get_weather_monthly_stats(latitude: float, longitude: float,
                              year: int, month: int):
    prev_month = month - 1
    prev_year = year
    if prev_month == 0:
        prev_month = 12
        prev_year = year - 1

    start_date = f"{prev_year}-{prev_month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={start_date}"
        f"&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,weathercode,"
        "relative_humidity_2m_max,relative_humidity_2m_mean,cloudcover_mean,"
        "shortwave_radiation_sum,snowfall_sum"
        "&timezone=Europe/Moscow"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Ошибка запроса: {response.status_code}, {response.text}"
        )

    data = response.json()
    daily = data.get('daily', {})

    df = pd.DataFrame({
        'temp_max': daily.get('temperature_2m_max', []),
        'temp_min': daily.get('temperature_2m_min', []),
        'precipitation': daily.get('precipitation_sum', []),
        'wind_speed_max': daily.get('wind_speed_10m_max', []),
        'weathercode': daily.get('weathercode', []),
        'humidity_max': daily.get('relative_humidity_2m_max', []),
        'humidity_mean': daily.get('relative_humidity_2m_mean', []),
        'cloudcover': daily.get('cloudcover_mean', []),
        'solar_radiation': daily.get('shortwave_radiation_sum', []),
        'snowfall': daily.get('snowfall_sum', [])
    })

    counts = {key: 0 for key in categories}
    for code in df['weathercode']:
        for category, codes in categories.items():
            if code in codes:
                counts[category] += 1
                break

    monthly_avg = df[
        ['temp_max', 'temp_min', 'precipitation', 'wind_speed_max',
         'humidity_max', 'humidity_mean', 'cloudcover', 'solar_radiation',
         'snowfall']
    ].mean().to_dict()

    result = {
        'monthly_avg': monthly_avg,
        'days_count': counts
    }

    return result


def create_new_weather_columns(
        df,
        max_retries=3,
        retry_delay=5,
        succes_delay=0.02,
        cache_path='denis/cache/weather_cache.pkl'
):
    cache = {}
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, 'rb') as f:
            print(f"Cache file '{cache_path}' успешно загружен.")
            cache = pickle.load(f)
    else:
        print(f"Cache file '{cache_path}' не найден или пустой, "
              "создаём новый кеш.")

    df_copy = df.copy()

    temp_max_list = []
    temp_min_list = []
    precipitation_list = []
    wind_speed_max_list = []

    humidity_max_list = []
    humidity_mean_list = []
    cloudcover_list = []
    solar_radiation_list = []
    snowfall_list = []

    category_counts = {cat: [] for cat in categories}

    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy),
                         desc="Processing rows"):
        try:
            state = row['state']
            year = int(row['year'])
            month = int(row['month'])
            key = (state, year, month)

            if key in cache:
                weather_stats = cache[key]
            else:
                latitude = regions_coords[state].get('latitude')
                longitude = regions_coords[state].get('longitude')

                for attempt in range(max_retries):
                    try:
                        weather_stats = get_weather_monthly_stats(
                            latitude, longitude, year, month
                        )
                        cache[key] = weather_stats
                        time.sleep(succes_delay)
                        break
                    except Exception as conn_err:
                        print(
                            f"Connection error on row {idx}, "
                            f"attempt {attempt+1}/{max_retries}: {conn_err}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            raise

            monthly_avg = weather_stats['monthly_avg']
            days_count = weather_stats['days_count']

            temp_max_list.append(monthly_avg.get('temp_max', None))
            temp_min_list.append(monthly_avg.get('temp_min', None))
            precipitation_list.append(monthly_avg.get('precipitation', None))
            wind_speed_max_list.append(monthly_avg.get('wind_speed_max', None))

            humidity_max_list.append(monthly_avg.get('humidity_max', None))
            humidity_mean_list.append(monthly_avg.get('humidity_mean', None))
            cloudcover_list.append(monthly_avg.get('cloudcover', None))
            solar_radiation_list.append(
                monthly_avg.get('solar_radiation', None))
            snowfall_list.append(monthly_avg.get('snowfall', None))

            for cat in categories:
                category_counts[cat].append(days_count.get(cat, 0))

        except Exception as e:
            print(f"Ошибка при обработке строки {idx}: {e}")
            temp_max_list.append(None)
            temp_min_list.append(None)
            precipitation_list.append(None)
            wind_speed_max_list.append(None)

            humidity_max_list.append(None)
            humidity_mean_list.append(None)
            cloudcover_list.append(None)
            solar_radiation_list.append(None)
            snowfall_list.append(None)

            for cat in categories:
                category_counts[cat].append(None)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {cache_path}")

    df_copy.loc[:, 'temp_max_avg'] = temp_max_list
    df_copy.loc[:, 'temp_min_avg'] = temp_min_list
    df_copy.loc[:, 'precipitation_sum'] = precipitation_list
    df_copy.loc[:, 'wind_speed_max_avg'] = wind_speed_max_list

    df_copy.loc[:, 'humidity_max_avg'] = humidity_max_list
    df_copy.loc[:, 'humidity_mean_avg'] = humidity_mean_list
    df_copy.loc[:, 'cloudcover_avg'] = cloudcover_list
    df_copy.loc[:, 'solar_radiation_sum'] = solar_radiation_list
    df_copy.loc[:, 'snowfall_sum'] = snowfall_list

    for cat in categories:
        df_copy.loc[:, f'days_{cat}'] = category_counts[cat]

    return df_copy




def expand_df_to_daily(
        df,
        regions_coords,
        max_retries=5,
        retry_delay=5,
        cache_path='denis/cache/weather_cache_daily.pkl'
    ):

    cache = {}
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, 'rb') as f:
            print(f"Cache file '{cache_path}' успешно загружен.")
            cache = pickle.load(f)
    else:
        print(f"Cache file '{cache_path}' не найден или пустой, создаётся новый кеш.")

    daily_dfs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Expanding months to days"):
        state = row['state']
        year = int(row['year'])
        month = int(row['month'])
        key = (state, year, month)

        if key in cache:
            daily_weather_df = cache[key]
        else:
            start_date = datetime.date(year, month, 1)
            last_day = calendar.monthrange(year, month)[1]
            end_date = datetime.date(year, month, last_day)

            start_str = start_date.isoformat()
            end_str = end_date.isoformat()

            latitude = regions_coords[state]['latitude']
            longitude = regions_coords[state]['longitude']

            success = False
            for attempt in range(max_retries):
                try:
                    daily_weather_df = get_weather_daily_stats(latitude, longitude, start_str, end_str)
                    cache[key] = daily_weather_df
                    success = True
                    time.sleep(0.5)
                    break
                except Exception as e:
                    print(f"Ошибка загрузки погоды для {state} {year}-{month}: {e}, попытка {attempt+1}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(cache, f)
                            print(f"Cache временно сохранён в {cache_path} перед выходом по ошибке.")
                        raise

            if not success:
                continue  # пропускаем текущий месяц

        # Добавляем колонки из исходной строки (кроме year/month/state, чтобы не перезаписать)
        for col in df.columns:
            if col not in ['year', 'month', 'state']:
                daily_weather_df[col] = row[col]

        daily_weather_df['state'] = state
        daily_weather_df['year'] = year
        daily_weather_df['month'] = month
        daily_dfs.append(daily_weather_df)

    # Сохраняем кеш после всех запросов
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {cache_path}")

    result_df = pd.concat(daily_dfs, ignore_index=True)
    return result_df



if __name__ == "__main__":
    from reader import df

    # print(categories)
    # df_upgraded = create_new_weather_columns(df)
    # print(df_upgraded)
    
    expanded_df = expand_df_to_daily(df, regions_coords)
    print(expanded_df.head(2))