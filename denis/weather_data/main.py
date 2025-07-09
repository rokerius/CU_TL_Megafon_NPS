import requests
import json
from config import weather_code_descriptions


def get_weather_open_meteo(latitude: float, longitud: float, date: str):
    """
    :param date: дата в формате 'YYYY-MM-DD'
    :return: словарь с данными о погоде (температура, осадки, код погоды, скорость ветра)
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={date}&end_date={date}"
        "&hourly=temperature_2m,precipitation,weathercode,wind_speed_10m"
        "&timezone=Europe/Moscow"
    )

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Ошибка запроса: {response.status_code}, {response.text}")
        


if __name__ == "__main__":
    latitude = 55.7558
    longitude = 37.6176
    date = "2023-05-01"

    weather_data = get_weather_open_meteo(latitude, longitude, date)
    print(json.dumps(weather_data, ensure_ascii=False, indent=4))

    code = 63 
    description = weather_code_descriptions.get(str(code), "Неизвестный код погоды")
    print(f"Код {code}: {description}")

