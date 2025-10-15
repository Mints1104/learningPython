import requests
import json 
import pandas as pd
from datetime import datetime

def process_forecast(data):
    forecast_list = []
    
    for forecast in data['list']:
        date_time_unix = forecast['dt']
        date_time = datetime.fromtimestamp(date_time_unix)
        temp = forecast['main']['temp']
        description = forecast['weather'][0]['description']
        
        forecast_list.append({
            'datetime': date_time,
            'temperature':temp,
            'description':description
        })
        df = pd.DataFrame(forecast_list)
        avg_temp = df['temperature'].mean()
        max_temp_row = df.loc[df['temperature'].idxmax()]
        print("\n-- Weather Analysis for Bristol ---")
        print(f"Average forecast temp: {avg_temp:.2f}°C")
        print(f"Highest temperature will be {max_temp_row['temperature']}°C on {max_temp_row['datetime']}")
        
        
    
def get_weather(api_key, lat, lon):
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    url = f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    print(f"Url: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")
        return None


api_key = "d8d2a750a493c1367f6d94809b5eeff3"
lat = 51.454514
lon = -2.587910
weather_data = get_weather(api_key, lat, lon)
if weather_data:
    process_forecast(weather_data)
else:
    print("Error")
    
