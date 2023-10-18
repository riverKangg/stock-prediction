import warnings
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Data collection
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
tic = yf.Ticker('O')
df = tic.history(period='1y')

# Data preprocessing
data = df.filter(['Close'])  # Select only the 'Close' column
dataset = data.values  # Convert to a NumPy array (251, 1)

# Set the length of training data
training_data_len = int(np.ceil(len(dataset) * 0.95))
print(training_data_len)

# Data scaling (in the range of 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create training data
train_data = scaled_data[0:int(training_data_len), :]

# Create input and output data
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print('x_train: ', x_train)
        print('y_train: ', y_train)
        print()
print(f'X_train:{len(x_train)}, y_train:{len(y_train)}')

# Convert to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the input data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create an LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create test data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert to NumPy arrays
x_test = np.array(x_test)

# Reshape the input data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the model
predictions = model.predict(x_test)

# Inverse scaling (back to the original values)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f'RMSE: {rmse}')

# Visualize the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()