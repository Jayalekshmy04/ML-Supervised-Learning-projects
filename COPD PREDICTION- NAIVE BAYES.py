import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from google.colab import files
files.upload()

copd=pd.read_csv('dataset.csv')
copd

copd.head()

copd.tail()

copd.shape

copd.describe()

copd.info()

copd.isnull().sum()

copd=copd.dropna()

copd.isnull().sum()

print(copd['COPDSEVERITY'].unique())

plt.figure(figsize=(5,3))
sns.countplot(x='COPDSEVERITY',data=copd)
plt.title('STAGES')
plt.show()


copd.replace({'COPDSEVERITY':{"SEVERE":0,"MODERATE":1,"VERY SEVERE":2,"MILD":3}},inplace=True)

copd['COPDSEVERITY'].value_counts()

x=copd.drop(columns='COPDSEVERITY')
y=copd['COPDSEVERITY']

print(x)

print(y)

correlation=copd.corr()

plt.figure(figsize=(5,4))
sns.heatmap(copd.corr())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print(x.shape,x_train.shape,x_test.shape)

print(x_train)

print(y_train)

x_test.to_csv('COPD_TEST.csv',index=False)
files.download('COPD_TEST.csv')

Scaler=StandardScaler()
Scaler.fit(x)

Standardized_data=Scaler.transform(x)
print(Standardized_data)

x=Standardized_data
y=copd['COPDSEVERITY']

print(x)

print(y)


Naive.Bayes

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

x_train_prediction=nb.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('accuracy on train prediction:',training_data_accuracy)
x_test_prediction=nb.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy on test prediction:',test_data_accuracy)


input_data=(85,92,72,75.0,492.0,440.0,492.0,0.94,30.0,2.47,60,22,13.0,45.3,3,3,1,2,1,0,0,0,0)
input_data_As_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_As_numpy_array.reshape(1,-1)
prediction=svm.predict(input_data_reshaped)
print(prediction[0])

y_test
