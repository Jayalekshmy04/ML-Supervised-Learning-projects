import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from google.colab import files
files.upload()

heart_data=pd.read_csv('heart.csv')
heart_data

heart_data.head()

heart_data.shape

heart_data.describe()

heart_data.isnull().sum()

heart_data['target'].value_counts()

x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, stratify=y, random_state=1)

print(x.shape, x_train.shape, x_test.shape)

print(x_train)

print(y_train)

model-training using logistic regression


model=LogisticRegression()

model.fit(x_train, y_train)
LogisticRegression()

model evaluation

x_train_prediction=model.predict(x_test)
training_data_accuracy=accuracy_score(x_train_prediction,y_test)

print('accuracy on training data:',training_data_accuracy)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction, y_test)

print('accuracy on test data:',test_data_accuracy)

input_data=(54,1,2,120,258,0,0,147,0,0.4,1,0,3)
input_data_as_numpy_array=np.array(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print("The person have heart disease")
else:
  print("The person does not have heart disease")