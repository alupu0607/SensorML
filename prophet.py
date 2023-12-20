import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

df = pd.read_csv('X:\\SensorMLDataset_small.csv')
labels = df.columns


def predict(start, end):
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


def get_cross_validation():
    models = predict(0, 1000)
    df_cv = []
    for model in models:
        df_cv += [
            cross_validation(model, initial='168 hours', period='168 hours', horizon='48 hours')]
    i = 1
    for df_cross_err in df_cv:
        df_p = performance_metrics(df_cross_err)
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
    get_cross_validation()


if __name__ == '__main__':
    main()
