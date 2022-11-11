import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data_set=pd.read_csv('Data.csv')
x=data_set.iloc[:,[0,1]].values
y=data_set.iloc[:,2].values
from Sklearn.model_selection import Train_Test_Split
x_Train,x_Test,y_Train,y_Test=Train_Test_Split(x,y,TextSize=0.25)

from Sklearn.Preproccesing import Standard_Scaler
st_x=Standard_Scaler()
x_Train=st_x.fit_transform(x_Train)
x_Test=st_x.transform(x_Test)

from Sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbour=5,matrix='mitsw',p=2)
Classifier.fit(x_Train,y_Train)
y_pred=Classifier.predict(x_test)

from Sklearn.matrix import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from Sklearn.matrix import Accuracy_Store
print("accuracy:",Accuracy_Store(y_test,y_pred))
print(cm)

 
