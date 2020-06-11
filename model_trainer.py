import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf

wholedata = pd.read_csv('all_stocks_5yr.csv')
print(wholedata.info())

print(len(wholedata['Name'].value_counts()))

"""converting date object in datetime format
and setting it as a index.
"""

wholedata['date'] = pd.to_datetime(wholedata['date'])
wholedata = wholedata.set_index('date')

"""encoding the Name columns in integer value and saving the encodings"""

labelencoder = LabelEncoder()
wholedata['Name']=labelencoder.fit_transform(wholedata['Name'].values)
label_mapping=dict(zip(labelencoder.classes_,labelencoder.transform(labelencoder.classes_)))

print("Label mapping for companies")
print(label_mapping)

"""AS this is timeseries prediction we will need the date column to be sorted in ascending order."""

wholedata = wholedata.sort_index(ascending=True, axis=0)

"""extracting apple data for visualisation and also the model would be train on this and will be use for other companies"""

data_AAPL = wholedata[wholedata['Name']==3]

"""extracting closing column data"""

data = data_AAPL.iloc[:, 3:4].values

"""Scaling data"""

scaler = MinMaxScaler()
scaled_data=scaler.fit_transform(data)

print(scaled_data.shape)

"""preparing data for time series prediction"""

X_train = []
Y_train = []
Timestamp = 100 #variable for timestep steps

length = len(scaled_data)

for i in range(Timestamp, length):
    X_train.append(scaled_data[i-Timestamp:i, 0])
    Y_train.append(scaled_data[i, 0])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train[0])
print()
print(Y_train[0])

print(X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)

layers = [tf.keras.layers.LSTM(92,return_sequences = True , input_shape = (X_train.shape[1],1)),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.LSTM(128,return_sequences = True),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.LSTM(128,return_sequences = True),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.LSTM(128, return_sequences = False),
         tf.keras.layers.Dense(units=1)]
model = tf.keras.models.Sequential(layers)

model.compile(optimizer='adam',loss = 'mean_squared_error' , metrics='accuracy')

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=4)

model.fit(X_train,Y_train,epochs=50,batch_size=32,callbacks=[callback],verbose=1)

"""saving model"""
model.save("model.h5")

company_label=int(input("Enter the company label no :"))
test_data = wholedata.loc[wholedata['Name']==company_label]
test_data = test_data.loc[:,test_data.columns == 'close']

# storing the actual stock prices in y_test starting from 100th day
y_test = test_data.iloc[Timestamp:, 0:].values

closing_price = test_data.iloc[:, 0:].values
closing_price_scaled = scaler.transform(closing_price)

x_test = []
length = len(test_data)

for i in range(Timestamp, length):
    x_test.append(closing_price_scaled[i-Timestamp:i, 0])

x_test = np.array(x_test)
x_test.shape

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape

"""predicting using model"""

y_pred = model.predict(x_test)
predicted_price = scaler.inverse_transform(y_pred)

"""ploting graph"""

plt.figure(figsize=(10,8))
plt.plot(y_test, color = 'blue', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
