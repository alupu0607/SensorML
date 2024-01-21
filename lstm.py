from operator import add

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import os

from risk_prediction import predict_risk
from tema1 import preprocess_data_IQR
from tema1 import load_data
def split_data(file_path):
    # The data is not randomly shuffled:
    # It ensures that chopping the data into windows of consecutive samples is still possible.
    df = pd.read_csv(file_path)
    date_time = pd.to_datetime(df.pop('Timestamp'),format='%m/%d/%Y %H:%M')
    print(df.shape[1]) # number of features (Timestamp is not a fetaure)
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    return train_df, val_df, test_df



class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

# Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=32,
      )

  ds = ds.map(self.split_window)

  return ds

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        result = next(iter(self.train))
        self._example = result
    return result


@property
def test_last_predictions(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_test_last_predictions', None)
    if result is None:
        result = next(iter(self.test))
        self._test_last_predictions = result
    return result


def compile_and_fit(model, window, patience=2):
  #early stopping prevents overfitting
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])

  mae_values = history.history['mean_absolute_error']
  return history, mae_values




def plot_future_train_set(self, model, plot_col, max_subplots=3, save_folder='train_set_predictions_lstm'):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))

    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        #labels_slice = slice(self.label_start, self.label_start + self.label_width + 24)
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)


        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions (Future)',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Timestamp')
    save_path = os.path.join(save_folder, f'{plot_col}.png')
    plt.savefig(save_path)
    plt.close()
def calculate_mae(actual_values, predicted_values):
    mae = mean_absolute_error(actual_values, predicted_values[:, -1])
    return mae

def plot_predictions_vs_actual(test_predictions, test_actual_values, param, save_folder='test_set_predictions_lstm'):
    shifted_predictions = np.roll(test_predictions[-1], 1)

    plt.figure(figsize=(12, 6))
    plt.plot(shifted_predictions, label='Predictions', marker='x', linestyle='--')
    plt.plot(test_actual_values[-24:], label='Actual Values', marker='o', linestyle='-')

    plt.xlabel('Time Step')
    plt.ylabel(param)
    plt.title(f'{param} Predictions vs Label Values')

    plt.legend()
    plt.grid(True)

    # Ensure that the save_folder directory exists
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, f'{param}.png')
    plt.savefig(save_path)
    plt.close()

MAX_EPOCHS = 20
WindowGenerator.split_window = split_window
WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
WindowGenerator.test_last_predictions = test_last_predictions
WindowGenerator.plot_future_train_set = plot_future_train_set
if __name__ == '__main__':

    file_path = "./SensorMLDataset.csv"

    train_df, val_df, test_df = split_data(file_path)
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    ## OUTLIER REMOVAL
    data = load_data(file_path)
    df = preprocess_data_IQR(data)

    ### Save the average MAE for every param
    average_mae_per_param = []

    params_to_track = ['umid', 'temp1', 'temp2']
    temp1 = []
    temp2 = []
    humidity = []


    for param in df.columns:
        if param != 'Timestamp':

            wide_window = WindowGenerator(
                #the model takes 24 consecutive observations as input
                #i am predicting the pres parameter 1 hour into the future
                #shift is 1 between consecutive windows
                input_width=24, label_width=24, shift=1, train_df= train_df,
                val_df=val_df, test_df=test_df,
                label_columns=[param])
            print(train_df.shape, val_df.shape, test_df.shape )


            lstm_model = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1)
            ])


            history, mae_values = compile_and_fit(lstm_model, wide_window)
            average_mae = np.mean(mae_values)
            print("Average Mean Absolute Error:", average_mae)
            average_mae_per_param.append(average_mae)


            if param in params_to_track:
                test_predictions = lstm_model.predict(wide_window.test)
                test_predictions = test_predictions * train_std[param] + train_mean[param]
                if param == 'temp1':
                    for value in test_predictions[-1]:
                        temp1 += [value[0]]
                elif param == 'temp2':
                    for value in test_predictions[-1]:
                        temp2 += [value[0]]
                else:
                    for value in test_predictions[-1]:
                        humidity += [value[0]]

            ### LAST PREDICTIONS for a 24 hours window, 1 hour into the future ###
            test_predictions = lstm_model.predict(wide_window.test)
            test_predictions = test_predictions * train_std[param] + train_mean[param]

            # This is that one actual prediction into the future
            print(test_predictions[-1])

            test_actual_values = wide_window.test_df[param].values * train_std[param] + train_mean[param]
            print(test_actual_values[-24:])

            plot_predictions_vs_actual(test_predictions, test_actual_values, param)

            wide_window.plot_future_train_set(lstm_model,param)
            #wide_window.plot_future_test_set(lstm_model,param,train_mean, train_std)

    air_array = list(map(add, temp1, temp2))
    air_array = [value / 2 for value in air_array]
    predict_risk(air_array, humidity, 'lstm')






