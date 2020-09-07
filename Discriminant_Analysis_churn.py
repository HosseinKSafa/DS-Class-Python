import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

'''
The telecom business is challenged by frequent customer churn due to 
several factors related to service and customer demographics. 
The dataset we'll use in our analysis includes a list of service-related 
factors about existing customers and information about whether they have stayed
or left the service provider.
Our objective is to predict which customers will potentially churn based
on service-related factors.


 The dataset consists of information for 7256 customers and includes 
 independent variables such as account length, number of voicemail messages,
 total daytime charge, total evening charge, total night charge, 
 total international charge, and number of customer service calls. 

The dependent variable in the dataset is whether the customer churned or not
'''

df=pd.read_csv('')

#Get the shape:
    
#Get the structure of variables:

#Get a summary:

#Missing values in predictors?

#Name of columns:

#Name of rows: 

#How many classes?

#proportions of classes in the response variable:

#Choose all columns except the last one and pu them in X:

#Choose the last column and call it y:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()
lr=LogisticRegression(penalty='none',solver='newton-cg')

lda.fit(X_train,y_train)
qda.fit(X_train,y_train)
gnb.fit(X_train,y_train)
lr.fit(X_train,y_train)

print('LDA:\n\n',confusion_matrix(y_test,lda.predict(X_test)),'\n')

print('QDA:\n\n',confusion_matrix(y_test,qda.predict(X_test)),'\n')

print('Naive Bayes:\n\n',confusion_matrix(y_test,gnb.predict(X_test)),'\n')

print('Logistc Regression:\n\n',confusion_matrix(y_test,lr.predict(X_test)),'\n')