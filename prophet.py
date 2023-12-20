import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

df = pd.read_csv('X:\\SensorMLDataset_small.csv')  # input the path to the CSV file here
labels = df.columns


def predict(start=0, end=168, week=0, echo=True):
    """Trains a Prophet model on the variables from the provided CSV file
        Parameters
        ----------
        start : int, optional
            The start row from the CSV file from which the training begins (default is 0)
        end : int, optional
            The end row from the CSV file up to which the model trains (default is 168, as the current CSV
            file has hourly data, meaning first week)
        week: int, optional
            Train on the data from the specified week (default is 0). If provided, it ignores start and end parameters
        echo: bool, optional
            Enables or disables prediction outputs (default is True)
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
            end = start + 169
    models = []  # for each column, train the model on that variable and add it to the list
    for label in labels[1:]:
        df_temp = df[[labels[0], label]]

        ts = df_temp[start:end]
        ts.columns = ['ds', 'y']

        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=48, freq='H')
        forecast = model.predict(future)

        if echo:
            forecast.rename(columns={"ds": "Timestamp", "yhat": "Predicted Value"}, inplace=True)
            print(f"----------- {label} ------------")
            print(forecast[['Timestamp', 'Predicted Value']].tail(48))

        models += [model]

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
    for df_cross_err in df_cv:
        df_p = performance_metrics(df_cross_err)
        if echo:
            print(f"---------{labels[i]}---------\n", df_p.tail(1))
        try:
            plot_cross_validation_metric(pd.DataFrame(df_cross_err), metric='mape').savefig(
                f"MAPE - {str(labels[i])}.png")
            i += 1
        except TypeError:
            plot_cross_validation_metric(pd.DataFrame(df_cross_err), metric='mae').savefig(
                f"MAE - {str(labels[i])}.png")
            i += 1


def main():
    predict(week=3)
    get_cross_validation()


if __name__ == '__main__':
    main()
