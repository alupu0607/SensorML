import pandas as pd
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

        models += [model]

    return models  # returnam lista cu modelele antrenate


def main():
    df = pd.read_csv('X:\\SensorMLDataset_small.csv')
    models = predict(df, 0, 1000)

    for model in models:
        df_cv = cross_validation(model, initial='168 hours', period='168 hours', horizon='48 hours')


if __name__ == '__main__':
    main()
