import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from google.colab import files
files.upload()

gold_data=pd.read_csv("gld_price_data.csv")
gold_data

gold_data.head()

gold_data.tail()

gold_data.shape

gold_data.info()

gold_data.isnull().sum()

gold_data.describe()

correlation

1.positive correlation


2.negative correlation


correlation=gold_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(gold_data.corr())

#correlation of gold values
#correlation =gold_data.corr()

#checking the distribution of the gold values
plt.figure(figsize=(24,6))
sns.displot(gold_data['GLD'],color='green')

x=gold_data.drop(['Date','EUR/USD'],axis=1)
y=gold_data['EUR/USD']

print(x)

print(y)

gold_data.value_counts('EUR/USD')

splitting into training and testing data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

print(x_train)

model training random forest regressor

regressor=RandomForestRegressor(n_estimators=100)

#training the model
regressor.fit(x_train,y_train)

model evaluation

#prediction on Testdata
test_data_prediction=regressor.predict(x_test)

print(test_data_prediction)

#R squared error
error_score=metrics.r2_score(y_test,test_data_prediction)
print("R squared:",error_score)

y_test=list(y_test)

plt.figure(figsize=(30,4))
plt.plot(y_test,color='blue',label='Actual values')
plt.plot(test_data_prediction,color='green',label='Predicted values')
plt.title('Actual Predicted Values')
plt.xlabel('Number of values')
plt.ylabel('gold values')
plt.legend()
plt.show()

input_data=(1373.199951,86.699997,71.849998,15.654)
input_data_as_numpy_array=np.array(input_data)
reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=regressor.predict(reshaped)
print('gold price:',prediction[0])