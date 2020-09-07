import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/men-track-records.csv')
X=df.iloc[:,:-1]
X=scale(X)
p=PCA()
p.fit(X)
W=p.components_.T
#Get the PC scores based on the centered X 
y=p.fit_transform(X)

#Compute the PC scores based on the original values of X (just for easier interpretation)

plt.figure(1)
#Get the scatter plot of the first two PC scores
plt.scatter(y[:,0],y[:,1],c="red",marker='o',alpha=0.5)


plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

#Put the name of the contries on the plotted datapoints (this is called annotation)
names=df['Country']


#names=names.agg(lambda x: x[:5])
for i, txt in enumerate(names):
    plt.annotate(txt, (y[i,0], y[i,1]))

#Get the first three columns of the matrix of loadings 
pd.DataFrame(W[:,:2],index=df.columns[:-1],columns=['W1','W2'])
#Compute the explained variability by the PC scores
pd.DataFrame(p.explained_variance_ratio_,index=range(1,9),columns=['Explained Variability'])
#Get the scree plot
plt.figure(2)
plt.bar(range(1,9),p.explained_variance_,color="blue",edgecolor="Red")
