import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


def feature_selection(df, correlation_threshold=0.8):
    numeric_columns = df.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()

    # Identify parameters with correlations above the threshold
    high_correlation_pairs = [(col1, col2, corr)
                              for col1 in corr_matrix.columns
                              for col2 in corr_matrix.columns
                              if col1 < col2 and abs(corr := corr_matrix.loc[col1, col2]) > correlation_threshold]

    # Create a set to store variables to drop
    variables_to_drop = set()

    for col1, col2, _ in high_correlation_pairs:
        # Add the variable with a higher number of missing values to the set
        if df[col1].isnull().sum() > df[col2].isnull().sum():
            variables_to_drop.add(col2)
        else:
            variables_to_drop.add(col1)

    # Drop the variables with high correlations and more missing values
    df.drop(variables_to_drop, axis=1, inplace=True)
    print(variables_to_drop)
    return df


df = pd.read_csv('SensorMLDataset_small.csv')  # input the path to the CSV file here
df=feature_selection(df)
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
    if not os.path.exists("predicted_vs_actual_plots"):
        os.makedirs("predicted_vs_actual_plots")

    for label in labels[1:]:
        df_temp = df[[labels[0], label]]

        if end + 48 > 1000:
            end -= 48
        actual_df = df_temp[start:(end + 48)]
        actual_df.columns = ['ds', 'y']

        ts = df_temp[start:end]
        ts.columns = ['ds', 'y']

        # Normalize the target with minmax
        scaler = MinMaxScaler()
        ts['y'] = scaler.fit_transform(ts['y'].values.reshape(-1, 1)).flatten()

        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=48, freq='H')
        forecast = model.predict(future)

        # Inverse normalized predictions to get the original scale to then compare with the actual values
        forecast['yhat'] = scaler.inverse_transform(forecast['yhat'].values.reshape(-1, 1)).flatten()

        file_path = f"predicted_vs_actual_plots/predicted_vs_actual_{label}_values.png"
        forecast.rename(columns={"ds": "Timestamp", "yhat": "Predicted Value"}, inplace=True)

        if echo:
            print(f"----------- {label} ------------")
            print(forecast[['Timestamp', 'Predicted Value']].tail(48))

        plt.figure(figsize=(10, 6))
        plt.plot(actual_df['ds'], actual_df['y'], label='Actual')
        plt.plot(forecast['Timestamp'], forecast['Predicted Value'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Normalized Value')  # Update ylabel to reflect normalization
        plt.title(f'Actual vs Predicted - {label}')
        plt.legend()
        plt.savefig(file_path)
        # if echo:
        #     plt.show()
        plt.close('all')

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
    predict(week=3)
    get_cross_validation()


if __name__ == '__main__':
    main()
