import json
import math
import numpy as np


def function(x, reference_array):  # vertex form of the quadratic function f(x) = a(x-h)^2 + k
    h = np.mean(reference_array)
    a = -100 / math.pow((max(reference_array) - h), 2)
    return a * math.pow((x - h), 2) + 100


def call_function(air_data, humidity_data, air_reference_interval, humidity_reference_interval):
    risk_sum_air = 0
    risk_sum_humidity = 0
    valid_terms_air = 0
    valid_terms_humidity = 0

    for air in air_data:
        if min(air_reference_interval) < air < max(air_reference_interval):
            risk_sum_air += function(air, air_reference_interval)
            valid_terms_air += 1
    for humidity in humidity_data:
        if min(humidity_reference_interval) < humidity < max(humidity_reference_interval):
            risk_sum_humidity += function(humidity, humidity_reference_interval)
            valid_terms_humidity += 1
    risk_avg_air = risk_sum_air / valid_terms_air if valid_terms_air != 0 else 0
    risk_avg_humidity = risk_sum_humidity / valid_terms_humidity if valid_terms_humidity != 0 else 0
    return (risk_avg_air + risk_avg_humidity) / 2


def predict_risk(air_data_array, humidity_data_array, model):
    EB_air_interval = [i for i in range(24, 30)]
    EB_humidity_interval = [i for i in range(90, 101)]

    GM_air_interval = [i for i in range(17, 24)]
    GM_humidity_interval = [i for i in range(90, 101)]

    LB_air_interval = [i for i in range(10, 25)]
    LB_humidity_interval = [i for i in range(90, 101)]

    LM_air_interval = [i for i in range(21, 24)]
    LM_humidity_interval = [i for i in range(85, 101)]

    PM_air_interval = [i for i in range(22, 30)]
    PM_humidity_interval = [i for i in range(50, 76)]

    dictionary = {
        "EarlyBlight": f"{call_function(air_data_array, humidity_data_array, EB_air_interval, EB_humidity_interval)}%",
        "GrayMold": f"{call_function(air_data_array, humidity_data_array, GM_air_interval, GM_humidity_interval)}%",
        "LateBlight": f"{call_function(air_data_array, humidity_data_array, LB_air_interval, LB_humidity_interval)}",
        "LateMold": f"{call_function(air_data_array, humidity_data_array, LM_air_interval, LM_humidity_interval)}%",
        "PowderyMildew": f"{call_function(air_data_array, humidity_data_array, PM_air_interval, PM_humidity_interval)}%"
    }
    with open(f'{model}.json', 'w') as json_file:
        json.dump(dictionary, json_file)
