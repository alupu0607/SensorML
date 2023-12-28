import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import os
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
      shuffle=True,
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

        labels_slice = slice(self.label_start, self.label_start + self.label_width + 24)
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)

        if model is not None:
            predictions = model(inputs)
            # Plot predictions for the future 24 hours
            plt.scatter(self.label_indices[-24:], predictions[n, -24:, label_col_index],
                        marker='X', edgecolors='k', label='Predictions (Future)',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Timestamp')
    save_path = os.path.join(save_folder, f'figure_{plot_col}.png')
    plt.savefig(save_path)
    plt.close()
def calculate_mae(actual_values, predicted_values):
    mae = mean_absolute_error(actual_values, predicted_values[:, -1])
    return mae
def create_actual_vs_predicted_list(actual_values, predicted_values):
    actual_vs_predicted = []

    for i in range(len(predicted_values)):
        actual = actual_values[i]
        predicted = predicted_values[i][-1]
        actual_vs_predicted.append((actual, predicted))

    return actual_vs_predicted


MAX_EPOCHS = 20
WindowGenerator.split_window = split_window
WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
WindowGenerator.plot_future_train_set = plot_future_train_set

if __name__ == '__main__':

    file_path = "./SensorMLDataset.csv"

    train_df, val_df, test_df = split_data(file_path)
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df = pd.read_csv(file_path)

    ### Save the average MAE for every param
    average_mae_per_param = []

    for param in df.columns:
        if param != 'Timestamp':

            wide_window = WindowGenerator(
                #the model takes 24 consecutive observations as input
                #i am predicting the pres parameter 1 hour into the future
                #shift is 1 between consecutive windows
                input_width=24, label_width=24, shift=1, train_df= train_df,
                val_df=val_df, test_df=test_df,
                label_columns=[param])
            print(wide_window)


            lstm_model = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1)
            ])
            #print('Input shape:', wide_window.example[0].shape)
            #print('Output shape:', lstm_model(wide_window.example[0]).shape)

            history, mae_values = compile_and_fit(lstm_model, wide_window)
            average_mae = np.mean(mae_values)
            print("Average Mean Absolute Error:", average_mae)
            average_mae_per_param.append(average_mae)

            val_performance = {}
            performance = {}
            IPython.display.clear_output()
            val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
            performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)



            ### TEST ACCURACY OF OBTAINED DATA - TO DO ###
            train_predictions = lstm_model.predict(wide_window.train)
            train_predictions = train_predictions * train_std[param] + train_mean[param]

            val_predictions = lstm_model.predict(wide_window.val)
            val_predictions = val_predictions * train_std[param] + train_mean[param]

            test_predictions = lstm_model.predict(wide_window.test)
            test_predictions = test_predictions * train_std[param] + train_mean[param]


            # De-normalize actual values - so as to show the last graphic with the next 1 hour prediction
            train_actual_values = wide_window.train_df[param].values * train_std[param] + train_mean[param]
            train_actual_values = train_actual_values[:-24]
            val_actual_values = wide_window.val_df[param].values * train_std[param] + train_mean[param]
            val_actual_values = val_actual_values[:-24]
            test_actual_values = wide_window.test_df[param].values * train_std[param] + train_mean[param]
            test_actual_values = test_actual_values[:-24]

            sets = {
                'Test': {'actual_values': test_actual_values, 'predicted_values': test_predictions},
                'Validation': {'actual_values': val_actual_values, 'predicted_values': val_predictions},
                'Eval': {'actual_values': test_actual_values, 'predicted_values': test_predictions}
            }

            for set_name, set_data in sets.items():
                actual_values = set_data['actual_values']
                predicted_values = set_data['predicted_values']


                #actual_vs_predicted = create_actual_vs_predicted_list(actual_values, predicted_values)

                #print(f"Actual versus Predicted Values for {set_name} Set:")
                #for actual, predicted in actual_vs_predicted:
                #    print(f"Actual: {actual}, Predicted: {predicted}")

            wide_window.plot_future_train_set(lstm_model,param)


            #### PRINT THE LAST HOUR PREDICTED INTO THE FUTURE (SCALED)





