from flask import Flask, jsonify
import numpy as np
import pandas as pd
import requests
from datetime import timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_model():
    
    response = requests.get("https://major-aytg.onrender.com/getdata")
    data = response.json()

    df = pd.DataFrame(data['response'])
    df['temperature'] = df['data'].apply(lambda x: str(int(float(x['temperature']))))
    df['humidity'] = df['data'].apply(lambda x: str(int(float(x['humidity']))))
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.sort_values(by='timestamp', inplace=True)

    scaler = MinMaxScaler()
    df[['temperature', 'humidity']] = scaler.fit_transform(df[['temperature', 'humidity']])

    sequence_length = 3
    X, y = [], []

    for i in range(len(df) - sequence_length):
        X.append(df[['temperature', 'humidity']].values[i:i+sequence_length])
        y.append(df[['temperature', 'humidity']].values[i+sequence_length])

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model_lstm.add(Dense(units=2))
    model_lstm.compile(optimizer='adam', loss='mse')
    history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    model_lr = LinearRegression()
    model_lr.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1))

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1))

    train_mse_lstm = model_lstm.evaluate(X_train, y_train, verbose=0)
    test_mse_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)

    future_timestamps = pd.date_range(df['timestamp'].iloc[-1] + timedelta(days=1), periods=2, freq='D')
    future_data = []

    for timestamp in future_timestamps:
        input_data = df[df['timestamp'] >= timestamp - timedelta(days=sequence_length)].tail(sequence_length)[['temperature', 'humidity']].values
        if len(input_data) < sequence_length:
            last_row = input_data[-1]
            while len(input_data) < sequence_length:
                input_data = np.vstack([input_data, last_row])
        future_data.append(input_data)

    future_data = np.array(future_data)
    future_data_scaled = scaler.transform(future_data.reshape(-1, 2)).reshape(-1, sequence_length, 2)
    
    future_predictions_scaled = model_lstm.predict(future_data_scaled)
    future_predictions = scaler.inverse_transform(future_predictions_scaled)

    future_predictions_df = pd.DataFrame(future_predictions, columns=['Future Temperature', 'Future Humidity'])
    future_predictions_df['Timestamp'] = future_timestamps
    future_predictions_df.set_index('Timestamp', inplace=True)

    future_predictions_df.index = future_predictions_df.index.strftime('%Y-%m-%d %H:%M:%S')
    future_predictions_json = future_predictions_df.to_dict(orient='index')

    training_mse = round(train_mse_lstm, 2)
    test_mse = round(test_mse_lstm, 2)
    training_accuracy = 1 - training_mse
    test_accuracy = 1 - test_mse

    training_accuracy_percent = round(training_accuracy * 100, 2)
    test_accuracy_percent = round(test_accuracy * 100, 2)

    train_mse_lr = mean_squared_error(y_train, model_lr.predict(X_train.reshape(X_train.shape[0], -1)))
    training_accuracy_lr = 1 - train_mse_lr
    training_accuracy_percent_lr = round(training_accuracy_lr * 100, 2)
    test_mse_lr = mean_squared_error(y_test, model_lr.predict(X_test.reshape(X_test.shape[0], -1)))
    testing_accuracy_lr = 1 - test_mse_lr
    testing_accuracy_percent_lr = round(testing_accuracy_lr * 100, 2)

    train_mse_rf = mean_squared_error(y_train, model_rf.predict(X_train.reshape(X_train.shape[0], -1)))
    training_accuracy_rf = 1 - train_mse_rf
    training_accuracy_percent_rf = round(training_accuracy_rf * 100, 2)
    test_mse_rf = mean_squared_error(y_test, model_rf.predict(X_test.reshape(X_test.shape[0], -1)))
    testing_accuracy_rf = 1 - test_mse_rf
    testing_accuracy_percent_rf = round(testing_accuracy_rf * 100, 2)

    print("\nLinear Regression Model:")
    print(f'Training Accuracy LR: {training_accuracy_percent_lr}')
    print(f'Testing Accuracy LR: {testing_accuracy_percent_lr}')

    print("\nRandom Forest Regressor Model:")
    print(f'Training Accuracy RF: {training_accuracy_percent_rf}')
    print(f'Testing Accuracy RF: {testing_accuracy_percent_rf}')

    print("\nLSTM Model:")
    print(f'Training Accuracy LSTM: {training_accuracy_percent}')
    print(f'Testing Accuracy LSTM: {test_accuracy_percent}')

    # plt.plot(history_lstm.history['loss'], label='LSTM Training MSE')
    # plt.plot(history_lstm.history['val_loss'], label='LSTM Validation MSE')
    # plt.title('Model Training and Validation MSE')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.show()

    return {
        "Training_MSE": training_mse,
        "Training_Accuracy": training_accuracy_percent,
        "Test_MSE": test_mse,
        "Test_Accuracy": test_accuracy_percent,
        "Future_Predictions": future_predictions_json
    }

@app.route('/run_model', methods=['GET'])
def run_model_endpoint():
    result = run_model()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
