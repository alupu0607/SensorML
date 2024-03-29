# General overview
"SensorMLDataset.csv" contains time-series data tracking the changes in various parameters such as temperature, light, humidity, and pressure, specifically for a tomato cultivation environment.

Our application predicts the evolution of environmental parameters for tomato cultivation using a combination of statistical and machine learning models:
- **Prophet**: A statistical model for forecasting time series data.
- **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) for modeling sequences with temporal dependencies.
- **Seq2Seq (Sequence to Sequence)**: A model for predicting future sequences of data points based on past sequences.

Aditionally, our application predicts the disease risk by using an aggregate formula which takes into account the air temperature and the air humidity.

<img width="838" alt="image" src="https://github.com/alupu0607/SensorML/assets/100222484/7dffae1e-0218-44f1-a66f-2dfd561ddba6">
<img width="838" alt="image" src="https://github.com/alupu0607/SensorML/assets/100222484/ee7dac3b-a111-4aca-8452-bb92921874d3">
<img width="838" alt="image" src="https://github.com/alupu0607/SensorML/assets/100222484/b846fa41-d3fb-4f67-af4b-09424f23802f">

# Implementation
We used Tensorflow and Python for the implementation.
- **The LSTM model** predicts 1 hour into the future by using a history of 24 hours; outliers were eliminated from the dataset before training
- **The Seq2Seq model** predicts 1 hour into th future by using a history of 24 hours; outliers were eliminated from the dataset before training.
-  **Prophet** predicts 48 hours into the future.

# Team Members

- Andreea Lupu: LSTM + Seq2Seq implementation
- Stefan Danila: Prophet analysis + risk formula
- Tudor Stroescu: Frontend

# References 
https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
