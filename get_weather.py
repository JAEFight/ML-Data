# https://github.com/open-meteo/sdk/tree/main
import openmeteo_requests
from openmeteo_sdk.Variable import Variable
from openmeteo_sdk.Aggregation import Aggregation
import pandas as pd
import numpy as np
def weather_values(city = "Wiesbaden",timespan = 1):
    diff_latitudes = {
        "Wiesbaden": {"latitude": 50.0826, "longitude": 8.2493},
        "Taunusstein": {"latitude":50.1499, "longitude":8.1521},
        "Berlin": {"latitude":52.5244, "longitude":13.4105}
    }
    om = openmeteo_requests.Client()
    params = {
        "latitude": diff_latitudes[city]["latitude"],
        "longitude": diff_latitudes[city]["longitude"],
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "rain", "snowfall", "rain_probability"],
        "current": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_mean", "uv_index_max"]
    }

    responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    #print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    #print(f"Elevation {response.Elevation()} m asl")
    #print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    #print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    # Current values
    current = response.Current()

    current_variables = list(map(lambda i: current.Variables(i), range(0, current.VariablesLength()))) #The Variable enumeration contains all available variables like temperature or wind_speed
    current_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, current_variables)) #temperature_2m is encoded as variable=temperature and altitude=2.
    current_relative_humidity_2m = next(filter(lambda x: x.Variable() == Variable.relative_humidity and x.Altitude() == 2, current_variables))

    #print(f"Current time {current.Time()}")
    #print(f"Current temperature_2m {current_temperature_2m.Value()}")
    #print(f"Current relative_humidity_2m {current_relative_humidity_2m.Value()}")

    #Daily Values - for daily aggregation is important  (7 days)
    daily = response.Daily()

    daily_variables = list(map(lambda i: daily.Variables(i), range(0, daily.VariablesLength())))
    #print(daily_variables)
    temperature_2m_max = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2 and x.Aggregation() == Aggregation.maximum, daily_variables)).ValuesAsNumpy()
    #print(f"Temperature_max {temperature_2m_max}")
    temperature_2m_min = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2 and x.Aggregation() == Aggregation.minimum, daily_variables)).ValuesAsNumpy()
    #print(f"Temperature_min {temperature_2m_min}")
    precipitation_probability_mean = next(filter(lambda x: x.Variable() == Variable.precipitation_probability and x.Aggregation() == Aggregation.mean, daily_variables )).ValuesAsNumpy()
    #print(f"Precipitation Probability : {precipitation_probability_mean}")
    uv_index_max = next(filter(lambda x: x.Variable() == Variable.uv_index and x.Aggregation() == Aggregation.maximum, daily_variables )).ValuesAsNumpy()
    #print(f"UV Index : {uv_index_max}")

    # Use Numpy for hourly (24hours x 7days)
    hourly = response.Hourly()
    hourly_time = range(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
    hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))

    hourly_temperature_2m = next(filter(lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, hourly_variables)).ValuesAsNumpy()
    hourly_precipitation = next(filter(lambda x: x.Variable() == Variable.precipitation, hourly_variables)).ValuesAsNumpy()
    hourly_wind_speed_10m = next(filter(lambda x: x.Variable() == Variable.wind_speed and x.Altitude() == 10, hourly_variables)).ValuesAsNumpy()

    rain = next(filter(lambda x: x.Variable() == Variable.rain, hourly_variables)).ValuesAsNumpy()
    rain_probability = next(filter(lambda x: x.Variable() == Variable.rain_probability, hourly_variables)).ValuesAsNumpy()

    #print(f"Rain sum of the day {rain}")
    #print(f"Rain Probability of the day {rain_probability}")

    """
    # Use Pandas to create a Dataframe out of the np. Arrays
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

    hourly_dataframe_pd = pd.DataFrame(data = hourly_data)
    print(hourly_dataframe_pd)
    """
    # Convert into correct format for the needed Timespan
    # Daily
    current_temperature = current_temperature_2m.Value()
    temperature_max = temperature_2m_max[0:timespan]
    temperature_min = temperature_2m_min[0:timespan]
    precipitation_prob = precipitation_probability_mean[0:timespan]
    uv_index_max = uv_index_max[0:timespan]
    wind_speed = hourly_wind_speed_10m[0:timespan*24]
    humidity_rel = current_relative_humidity_2m.Value()
        

    temperature = temperature_max, temperature_min, current_temperature
    weather_dict = {
        "temperature": temperature,
        "relative_humidity": humidity_rel,
        "wind_speed": wind_speed,
        "uv-index": uv_index_max,
        "precipitation_prob": precipitation_prob
    }
    text = f"""Hey, heute liegt die Temperatur aktuell bei {round(current_temperature,2)}°C, 
    mit einem Maximum von {round(float(temperature_max[0]),2)}°C und einem Minimum von {round(float(temperature_min[0]),2)}°C.
    Die Luftfeuchtigkeit liegt bei {int(humidity_rel)}%, die heutige Windgeschwindigkeit beträgt {int(np.median(wind_speed))} 
    km/h, der UV-Index ist bei {round(float(uv_index_max[0]),2)}, also denk an Sonnenschutz, 
    und die Regenwahrscheinlichkeit beträgt {round(float(precipitation_prob[0]),2)}%!"""


    return weather_dict, text

#weather_stats = weather_values("Wiesbaden",1)

"""
temp = weather_stats["temperature"]
humidity = weather_stats["relative_humidity"]
wind_speed = weather_stats["wind_speed"]
uv = weather_stats["uv-index"]
prec_prob = weather_stats["precipitation_prob"]
"""
