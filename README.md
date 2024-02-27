# About
"SensorMLDataset.csv" contains time-series data tracking the changes in various parameters such as temperature, light, humidity, and pressure, specifically for a tomato cultivation environment.
We used Tensorflow and Python for the implementation.


Our application predicts the evolution of environmental parameters for tomato cultivation using a combination of statistical and machine learning models:
- **Prophet**: A statistical model for forecasting time series data.
- **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) for modeling sequences with temporal dependencies.
- **Seq2Seq (Sequence to Sequence)**: A model for predicting future sequences of data points based on past sequences.

Aditionally, our application predicts the disease risk by using an aggregate formula which takes into account the air temperature and the air humidity.

![LSTM](https://imgur.com/a/QF9WBUA)

# Team Members

- Andreea Lupu: LSTM + Seq2Seq implementation
- Stefan Danila: Prophet analysis + risk formula
- Tudor Stroescu: Frontend 
