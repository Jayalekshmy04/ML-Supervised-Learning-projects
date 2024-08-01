import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from google.colab import files
files.upload()

stellar=pd.read_csv('star_classification.csv')
stellar

stellar.head()

stellar.tail()

stellar.shape

stellar.describe()

stellar.info()

stellar.isnull().sum()

stellar['cam_col'].value_counts()

stellar['class'].unique()

plt.figure(figsize=(4,4))
sns.countplot(x='class',data=stellar)
plt.title('CLASSIFICATION')
plt.show()

plt.figure(figsize=(4,4))
sns.distplot(stellar['cam_col'])
plt.show()

plt.figure(figsize=(4,4))
sns.histplot(stellar['delta'])
plt.title('DELTA')
plt.show()

correlation=stellar.corr()


plt.figure(figsize=(11,11))
sns.heatmap(correlation,annot=True)

stellar.replace({'class':{'GALAXY':0, 'QSO':1, 'STAR':2}}, inplace=True)

x=stellar.drop(columns=['class','obj_ID','spec_obj_ID'])
y=stellar['class']

print(x)

print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)

x.shape,x_train.shape,x_test.shape

print(x_train)

print(y_train)

x_test.to_csv('stellar111.csv',index=False)
files.download('stellar111.csv')

scaler=StandardScaler()
scaler.fit(x)

standardized_data=scaler.transform(x)
standardized_data

x=standardized_data
y=stellar['class']

print(x)

print(y)

RF=RandomForestClassifier()
RF.fit(x_train,y_train)

train_data_prediction=RF.predict(x_train)
training_data_accuracy=accuracy_score(y_train,train_data_prediction)
print('accuracy on training data',training_data_accuracy)
test_data_prediction=RF.predict(x_test)
test_data_accuracy=accuracy_score(y_test,test_data_prediction)
print('accuracy on test data',test_data_accuracy)

KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)

train_data_prediction=KNN.predict(x_train)
training_data_accuracy=accuracy_score(y_train,train_data_prediction)
print('accuracy on training data',training_data_accuracy)
test_data_prediction=KNN.predict(x_test)
test_data_accuracy=accuracy_score(y_test,test_data_prediction)
print('accuracy on test data',test_data_accuracy)

DTC=DecisionTreeClassifier()
DTC.fit(x_train,y_train)

train_data_prediction=DTC.predict(x_train)
training_data_accuracy=accuracy_score(y_train,train_data_prediction)
print('accuracy on training data',training_data_accuracy)
test_data_prediction=DTC.predict(x_test)
test_data_accuracy=accuracy_score(y_test,test_data_prediction)
print('accuracy on test data',test_data_accuracy)

input_data=(116.796713508914,27.6222136013817,18.23172,17.26519,16.98849,16.86762,16.80546,2864,301,6,58,0.0008899521,2055,53729,176)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
RF_prediction=RF.predict(input_data_reshaped)
print(RF_prediction)

y_test