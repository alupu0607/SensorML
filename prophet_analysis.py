import pandas as pd
from matplotlib import pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation


def predict(df, start, end):
    labels = df.columns
    df[labels[0]] = pd.to_datetime(df[labels[0]], format='%m/%d/%Y %H:%M')

    models = []  # pentru fiecare parametru, adaugam modelul antrenat intr-o lista
    for label in labels[1:]:
        df_temp = df[[labels[0], label]]
        print(f"----------- {label} ------------")
        ts = df_temp[start:end]
        ts.columns = ['ds', 'y']

        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=48, freq='H')
        forecast = model.predict(future)
        print(forecast[['ds', 'yhat']].tail(48))

        # Adăugăm coloanele 'yhat' (valoarea prezisă) și 'y' (valoarea reală)
        df_pred = forecast[['ds', 'yhat']].set_index('ds')
        df_real = ts.set_index('ds')
        df_compare = df_real.join(df_pred, how='outer')

        file_path = f"predicted_vs_actual_plots/predicted_vs_actual_{label}_values.png"

        plt.figure(figsize=(12, 6))
        plt.plot(df_compare.index, df_compare['y'], label='Actual')
        plt.plot(df_compare.index, df_compare['yhat'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel(label)
        plt.title(f'Predicted vs Actual {label} Values')
        plt.legend()
        plt.savefig(file_path)
        plt.show()
        plt.close()

        models += [model]

    return models


def main():
    df = pd.read_csv('SensorMLDataset_small.csv')
    models = predict(df, 0, 1000)

    for model in models:
        df_cv = cross_validation(model, initial='168 hours', period='168 hours', horizon='48 hours')


if __name__ == '__main__':
    main()
