# Time-Series-Prediction-of-Airline-Passengers-Using-LSTM-Networks
Description:
This project focuses on predicting the number of airline passengers using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) well-suited for time series forecasting. The dataset used contains monthly totals of international airline passengers from 1949 to 1960.

1. Data Preparation:
The dataset is loaded from a CSV file, and only the passenger count column is used. The data is normalized to the range [0, 1] using MinMaxScaler to enhance the performance and convergence of the neural network. The data is then split into training and testing sets, with approximately 67% of the data used for training and the remaining for testing.

2. Data Transformation:
To prepare the data for the LSTM network, the time series data is transformed into a supervised learning problem using a sliding window approach. The function create_dataset is defined to generate sequences of previous observations (look_back) to predict the next observation. The input data is reshaped to the format [samples, time steps, features] required by LSTM layers.

3. Model Building and Training:
A Sequential LSTM model is built with the following architecture:

An LSTM layer with 4 units and an input shape based on the look_back value.
A Dense layer with a single unit to output the predicted passenger count.
The model is compiled using the Adam optimizer and mean squared error loss function. It is trained on the training data for 100 epochs with a batch size of 1.

4. Prediction and Evaluation:
After training, the model is used to make predictions on both the training and testing datasets. These predictions are then inverted to the original scale using the scaler. The model's performance is evaluated using Root Mean Squared Error (RMSE) for both training and testing sets.

5. Visualization:
The actual passenger counts and the model's predictions are plotted to visualize the model's performance. The predictions are shifted to align with the actual data for clear comparison.

6. Experimentation with Different Look-Back Values:
The model is further tested with different look-back values (1, 3, and 5) to observe the impact of the length of historical data used for predictions. For each look-back value, the model is retrained, and the RMSE for both training and testing sets is calculated and reported.

By leveraging LSTM networks, this project effectively demonstrates time series forecasting, providing insights into the trends and patterns in the airline passenger data. The experimentation with different look-back values also highlights the importance of choosing appropriate historical data lengths for accurate predictions.
