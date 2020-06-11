import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading model
model = load_model('model.h5')

#loading data
wholedata = pd.read_csv('all_stocks_5yr.csv')
wholedata['date'] = pd.to_datetime(wholedata['date'])
wholedata = wholedata.set_index('date')
labelencoder = LabelEncoder()
wholedata['Name']=labelencoder.fit_transform(wholedata['Name'].values)
label_mapping=dict(zip(labelencoder.classes_,labelencoder.transform(labelencoder.classes_)))

print(label_mapping)
print("choose the label for corresponding company from the above list")


company_label=int(input("Enter the company label no :"))
#getting the key of label ie. company name from label
company_name=(list(label_mapping.keys())[list(label_mapping.values()).index(company_label)])
print("you chose {} company".format(company_name))
com_data = wholedata.loc[wholedata['Name']==company_label]
com_data = com_data.loc[:,com_data.columns == 'close']

scaler = MinMaxScaler()
Timestamp=100

# storing the actual stock prices in y_test starting from 100th day
y_com = com_data.iloc[Timestamp:, 0:].values

closing_price = com_data.iloc[:, 0:].values
closing_price_scaled = scaler.fit_transform(closing_price)

x_com = []
length = len(com_data)

for i in range(Timestamp, length):
    x_com.append(closing_price_scaled[i-Timestamp:i, 0])

x_com = np.array(x_com)

x_com = np.reshape(x_com, (x_com.shape[0], x_com.shape[1], 1))


"""predicting using model"""

y_pred = model.predict(x_com)
predicted_price = scaler.inverse_transform(y_pred)

"""ploting graph"""

plt.figure(figsize=(10,8))
plt.plot(y_com, color = 'blue', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction on {}'.format(company_name))
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
