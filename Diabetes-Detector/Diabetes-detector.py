"""
Created on Fri Aug 31 21:25:16 2018

@author: vignesh
"""
#Importing the dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
#Reading data

data = pd.read_csv('diabetes.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,8].values

#Feature Scaling

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

#Defining

model = Sequential()
model.add(Dense(16,input_shape=(8,),activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))

#compiling
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Fitting

model.fit(x_train,y_train,batch_size=20,epochs=100,verbose=1)

#Metrics

y_pred = model.predict_classes(x_test)
metrics = model.evaluate(x_test,y_test)
matrix = confusion_matrix(y_test,y_pred)
print('Loss: {}\nAccuracy: {}'.format(*metrics))

#Saving the model

model.save('trained_diabetes_detector.h5')








