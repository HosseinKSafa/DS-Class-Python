import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score



df=pd.read_csv("OnlineNewsPopularity-c.csv")

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X=pd.DataFrame(scale(X))



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


#Define the models

#Linear
svm_linear=svm.SVC(C=10,kernel="linear")

#Radial basis function
svm_rbf=svm.SVC(C=100,kernel="rbf",gamma=0.001)
#gamma is 1/sigma^2


#Polynomial kernel
svm_poly=svm.SVC(C=100,kernel="poly",degree=2,gamma=0.001)


#Fit the models
svm_linear.fit(X_train,y_train)

svm_rbf.fit(X_train,y_train)

svm_poly.fit(X_train,y_train)

print('Linear','\n',confusion_matrix(y_test,svm_linear.predict(X_test)))
print('RBF','\n',confusion_matrix(y_test,svm_rbf.predict(X_test)))
print('Polynomial','\n',confusion_matrix(y_test,svm_poly.predict(X_test)))

'''
import pickle
#Save the model
file_name= "svm_news.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(svm_rbf, file)
#Read the model and use it in future
with open(file_name, 'rb') as file:
    pickle_model = pickle.load(file)
'''














    
