import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df=pd.read_csv('.../airqualityteh.csv')
df.dropna(inplace=True)
df.index=range(df.shape[0])
X=df.iloc[:,2:]
scaledX=scale(X)
p=PCA()
p.fit(scaledX)
W=p.components_.T
#Get the PC scores based on the centered X 
y=p.fit_transform(scaledX)



#make a list of colors for different month
col=['r', 'brown', 'g', 'black', 'gray', 'cyan', 'blue', 'deeppink', 'yellow']

#Assign the colors to the observations. For instance, all observations in Ordibehesht will be red
#in the scatter plot

#define a function that can get the number of month and choose a color from the list of color defined
#in col
def choosecol(x):# x is the number of month (it starts from 2 and ends in 10)
    return(col[x-2])#choose a color from the list. since the list starts from zero, we need to 
    # have x-2 (months start from 2 and the list indices start from zero)
#for example, if we want to select a color from col for the 4th month, we can write choosecol(4),
#and it will return 'g'      

#now we can apply the choosecol function to all elements of df['month'] to assign the colors to 
#the monthes
pointcolors=df['month'].agg(choosecol)
#pointcolors is the vector of assigned colors


xs=y[:,0]#xs represents PC score 1
ys=y[:,1]#ys represents PC score 2
#make a scatter plot of the first two PC scores. The argument c is the color of the poiints
plt.scatter(y[:,0],y[:,1],c=pointcolors,marker='o')
plt.ylim([-4,9])
#The following will give you the bplot
#plot the arrows associated with variables
for i in range(len(W[:,0])):
#arrows project features (ie columns from csv) as vectors onto PC axes
#here we multiply W by abs(max(xs)) and abs(max(ys)) to scale the biplots
#the plt.arrow function draws some arrows for us. The biplot is nothing but 
#some arrows specifying the variables in the scatter plots    
# The arrows need a starting point defined by np.mean(xs) and np.mean(ys) and an ending point.
#the end point for the ith variable in our biplot should be defined as W[i,0]*abs(max(xs)) and W[i,1]*abs(max(ys))
    plt.arrow(np.mean(xs), np.mean(ys), W[i,0]*abs(max(xs)), W[i,1]*abs(max(ys)),
              color='black', width=0.0005, head_width=0.0025)
    plt.text(W[i,0]*abs(max(xs))+np.mean(xs), +np.mean(ys)+W[i,1]*abs(max(ys)),
             list(X.columns.values)[i], color='black')

