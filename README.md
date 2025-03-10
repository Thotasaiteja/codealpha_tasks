import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import yfinance as yf
data=yf.download('AAPL', start='2010-01-01', end='2025-02-02', progress=False)
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
df = data[['Date', 'Close']]
df.set_index('Date', inplace=True)
df.dropna(inplace=True)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
def create_xy(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:i+n_steps, 0])
        y.append(data[i+n_steps, 0])
    return np.array(x), np.array(y)
n_steps = 60
x_train, y_train = create_xy(train_data, n_steps)
x_test, y_test = create_xy(test_data, n_steps)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=20)  # Increase epochs for better results
predictions = model.predict(x_test)
y_test_inv = scaler.inverse_transform(np.reshape(y_test, (-1, 1)))
predictions_inv = scaler.inverse_transform(np.reshape(predictions, (-1, 1)))
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='Actual Price')
plt.plot(predictions_inv, label='Predicted Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
