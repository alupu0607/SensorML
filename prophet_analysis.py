import json
from operator import add

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from tema1 import preprocess_data_IQR
from tema1 import load_data
from risk_prediction import predict_risk
df = load_data('SensorMLDataset.csv')  # input the path to the CSV file here

# # OUTLIER REMOVAL
# df = preprocess_data_IQR(df)
labels = df.columns


def predict(start=0, end=15913, week=0, echo=True):
    """Trains a Prophet model on the variables from the provided CSV file
        Parameters
        ----------
        start : int, optional
            The start row from the CSV file from which the training begins (default is 0)
        end : int, optional
            The end row from the CSV file up to which the model trains (default is 15913, as the current CSV
            file has hourly data, meaning up until last hour)
        week: int, optional
            Train on the data from the specified week (default is 0). If provided, it ignores start and end parameters
        echo: bool, optional
            Enables or disables prediction outputs, as well as display of predicted_vs_actual plots (default is True)
        Returns
        -------
        list
            a list of the models trained on the variables from the CSV file
        """
    df[labels[0]] = pd.to_datetime(df[labels[0]], format='%m/%d/%Y %H:%M')
    if week - 1 > df.__len__() // 168:
        print("Not enough weeks in the dataset. Defaulting to start, end parameters.")
    else:
        if week != 0:
            start = (week - 1) * 168
            end = (start + 169) % 1001
    models = []  # for each column, train the model on that variable and add it to the list
    air_array = []
    air_temp1_array = []
    air_temp2_array = []
    humidity_array = []
    if not os.path.exists("predicted_vs_actual_plots"):
        os.makedirs("predicted_vs_actual_plots")

    for label in labels[1:]:
        df_temp = df[[labels[0], label]]

        ts = df_temp[start:end]
        ts.columns = ['ds', 'y']

        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=48, freq='H')
        forecast = model.predict(future)

        file_path = f"predicted_vs_actual_plots/{label}.png"

        if label == 'temp1':
            for value in forecast['yhat'].tail(48):
                air_temp1_array += [value]
        if label == 'temp2':
            for value in forecast['yhat'].tail(48):
                air_temp2_array += [value]
        if label == 'umid':
            for value in forecast['yhat'].tail(48):
                humidity_array += [value]
        if label in ["lumina", "B500", "G550", "O600", "R650", "V450", "Y570"]:
            for i in range(len(forecast['yhat'])):
                if forecast['yhat'][i] < 0:
                    forecast['yhat'][i] = 0
        forecast.rename(columns={"ds": "Timestamp", "yhat": "Predicted Value"}, inplace=True)
        if echo:
            print(f"----------- {label} ------------")
            print(forecast[['Timestamp', 'Predicted Value']].tail(48))

        plt.figure(figsize=(10, 6))
        plt.plot(ts['ds'].tail(96), ts['y'].tail(96), label='Actual')
        plt.plot(forecast['Timestamp'].tail(48), forecast['Predicted Value'].tail(48), label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Actual vs Predicted - {label}')
        plt.legend()
        plt.savefig(file_path)

        plt.close()

        models += [model]
    air_array = list(map(add, air_temp1_array, air_temp2_array))
    air_array = [value / 2 for value in air_array]

    predict_risk(air_array, humidity_array, 'prophet')
    return models  # return trained models


def get_cross_validation(echo=True):
    """Computes the error at training using time-series cross validation on the entire dataset.
    It outputs the calculated error plots for each variable as a .PNG file
    Parameters
    ----------
        echo: bool, optional
            Enables or disables computed errors outputs (default is True)
    """
    models = predict(0, df.__len__(), echo=False)
    print("----- Computing time series cross validation -----")
    df_cv = []
    for model in models:
        df_cv += [
            cross_validation(model, initial='168 hours', period='168 hours', horizon='48 hours')]
    i = 1
    if not os.path.exists("cross_validation_values"):
        os.makedirs("cross_validation_values")
    for df_cross_err in df_cv:
        df_p = performance_metrics(df_cross_err)

        if echo:
            print(f"---------{labels[i]}---------\n", df_p.tail(1))

        # Check if MAPE is available, otherwise use MAE
        metric_to_use = 'mape' if 'mape' in df_p.columns else 'mae'

        plot_cross_validation_metric(pd.DataFrame(df_cross_err), metric=metric_to_use).savefig(
            f"cross_validation_values/{metric_to_use.upper()} - {str(labels[i])}.png")

        i += 1


def main():
    predict()
    # get_cross_validation()


if __name__ == '__main__':
    main()
