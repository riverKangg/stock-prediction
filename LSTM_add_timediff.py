import warnings
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
# from sklearn preprocessing import MinMaxScaler
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the start and end dates for data retrieval
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Fetch historical stock data for 'O' using yfinance
tic = yf.Ticker('O')
df = tic.history(period='1y')

# Calculate the time interval between data points in seconds
df['time_interval'] = list(df.index)
df['time_interval'] = df['time_interval'].apply(lambda x: x.to_pydatetime())
df['time_interval'] = df['time_interval'].diff().dt.total_seconds()

# Select relevant columns for analysis
data = df.filter(['Close', 'time_interval'])
dataset = data.values  # Shape: (251, 2)

# Determine the length of the training dataset
training_data_len = int(np.ceil(len(dataset) * 0.95))  # 239

# Copy the dataset for scaling
scaled_data = dataset[:]

# Create training data
train_data = scaled_data[0:int(training_data_len), :]

# Initialize lists to store input (x) and output (y) training data
x_train, y_train = [], []

# Define the sequence length (number of time steps to consider)
sequence_length = 60

# Create input sequences and corresponding output values
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i - sequence_length:i, :])
    y_train.append(train_data[i, 0])

# Convert lists to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the input data to fit the model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
print(x_train.shape[0], x_train.shape[1], x_train.shape[2])  # Expected output: 179 60 2

# Create an LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare test data
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i - sequence_length:i, :])

# Convert the test data to NumPy array
x_test = np.array(x_test)

# Reshape the test data to match the model input shape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
predictions = model.predict(x_test)
# Inverse scaling (if using MinMaxScaler)
# predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f'RMSE: {rmse}')