# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:29:27 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("glass.csv")

df.dtypes 
df.head()
df.shape

# Blanks
df.isnull()
df.isnull().sum()
list(df)
# finding duplicated rows
df.duplicated() # hence no duplicated rows

# finding duplicated columns
df.columns.duplicated() # Hence no duplicated columns
#==================================================
# Data visualization
import seaborn as sns
sns.pairplot(df)
sns.heatmap(df)

#===================
df.boxplot(column="RI",vert=False)
 

import numpy as np
Q1 = np.percentile(df["RI"],25)
Q2 = np.percentile(df["RI"],50)
Q3 = np.percentile(df["RI"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["RI"]<LW) | (df["RI"]>UW)]
len(df[(df["RI"]<LW) | (df["RI"]>UW)])
# there are 8 outlaiers
#===================
df.boxplot(column="Na",vert=False)
 

import numpy as np
Q1 = np.percentile(df["Na"],25)
Q2 = np.percentile(df["Na"],50)
Q3 = np.percentile(df["Na"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Na"]<LW) | (df["Na"]>UW)]
len(df[(df["Na"]<LW) | (df["Na"]>UW)])
# there are 1 outlaier
#===========================
df.boxplot(column="Mg",vert=False)
 

import numpy as np
Q1 = np.percentile(df["Mg"],25)
Q2 = np.percentile(df["Mg"],50)
Q3 = np.percentile(df["Mg"],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Mg"]<LW) | (df["Mg"]>UW)]
len(df[(df["Mg"]<LW) | (df["Mg"]>UW)])
# there are no outlaiers

df.boxplot(column="Al",vert=False)
 

import numpy as np
Q1 = np.percentile(df["Al"],25)
Q2 = np.percentile(df["Al"],50)
Q3 = np.percentile(df["Al"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Al"]<LW) | (df["Al"]>UW)]
len(df[(df["Al"]<LW) | (df["Al"]>UW)])
# in this variable 6 outlaiers

df.boxplot(column="Si",vert=False)
 

import numpy as np
Q1 = np.percentile(df["Si"],25)
Q2 = np.percentile(df["Si"],50)
Q3 = np.percentile(df["Si"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Si"]<LW) | (df["Si"]>UW)]
len(df[(df["Si"]<LW) | (df["Si"]>UW)])
# there are 6 outlaiers

df.boxplot(column="K",vert=False)
 

import numpy as np
Q1 = np.percentile(df["K"],25)
Q2 = np.percentile(df["K"],50)
Q3 = np.percentile(df["K"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["K"]<LW) | (df["K"]>UW)]
len(df[(df["K"]<LW) | (df["K"]>UW)])
# there are 6 outlaiers

df.boxplot(column="Ca",vert=False)
 

import numpy as np
Q1 = np.percentile(df["Ca"],25)
Q2 = np.percentile(df["Ca"],50)
Q3 = np.percentile(df["Ca"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Ca"]<LW) | (df["Ca"]>UW)]
len(df[(df["Ca"]<LW) | (df["Ca"]>UW)])
# out layers 16
df["Ca"]=np.where(df["Ca"]>UW,UW,np.where(df["Ca"]<LW,LW,df["Ca"]))
df[(df["Ca"]<LW) | (df["Ca"]>UW)]
len(df[(df["Ca"]<LW) | (df["Ca"]>UW)])
# there are no outlaiers
#
df.boxplot(column="Ba",vert=False)


import numpy as np
Q1 = np.percentile(df["Ba"],25)
Q2 = np.percentile(df["Ba"],50)
Q3 = np.percentile(df["Ba"],75)
IQR = Q3 - Q1
LW = Q1 - (3.5*IQR)
UW = Q3 + (3.5*IQR)
df[(df["Ba"]<LW) | (df["Ba"]>UW)]
len(df[(df["Ba"]<LW) | (df["Ba"]>UW)])
# out layers 38
df["Ba"]=np.where(df["Ba"]>UW,UW,np.where(df["Ba"]<LW,LW,df["Ba"]))
df[(df["Ba"]<LW) | (df["Ba"]>UW)]
len(df[(df["Ba"]<LW) | (df["Ba"]>UW)])
# outlaiers are 0
#


df.boxplot(column="Fe",vert=False)

import numpy as np
Q1 = np.percentile(df["Fe"],25)
Q2 = np.percentile(df["Fe"],50)
Q3 = np.percentile(df["Fe"],75)
IQR = Q3 - Q1
LW = Q1 - (2.5*IQR)
UW = Q3 + (2.5*IQR)
df[(df["Fe"]<LW) | (df["Fe"]>UW)]
len(df[(df["Fe"]<LW) | (df["Fe"]>UW)])
# outliers are 2

#===================================================================================
# splitting the variables
X = df.iloc[:,0:9]
Y = df.iloc[:,9]

#====================================================================================
# dropping the outlayres from the data set
X = X.drop([47,103,110,111,112,131,106,107,184,163,171,172,192,208,209,188,201,162,174])
Y = Y.drop([47,103,110,111,112,131,106,107,184,163,171,172,192,208,209,188,201,162,174])
X.shape
Y.shape
#======================================================================================

import matplotlib.pyplot as plt
plt.scatter(X["RI"],X["Na"],color = "black")
plt.show()
plt.figure(figsize=(30,7))

plt.scatter(X["RI"],X["Mg"],color = "black")
plt.show()


#========================================================================= 
# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)
X_scale = pd.DataFrame(X_scale)

#==========================================================================
# splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,test_size=0.30,random_state=12)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#===========================================================================

from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
testing_accuracy = []

neighbors = range(1,11)

for number_of_neighbors in neighbors:
    KNN=KNeighborsClassifier(n_neighbors=number_of_neighbors)
    KNN.fit(X_train,Y_train)
    training_accuracy.append(KNN.score(X_train,Y_train))
    testing_accuracy.append(KNN.score(X_test,Y_test))

print(training_accuracy)    
print(testing_accuracy)

import matplotlib.pyplot as plt

plt.plot(neighbors,training_accuracy,label="training accuracy")
plt.plot(neighbors,testing_accuracy,label="testing accuracy")
plt.ylabel("Accuracy")
plt.slabel("number of neighbors")
plt.legend()
# therefore by seeing the plot i deside that K=6 the best value
#============================================================================
# Model fitting
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=6,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#=======================================================================
# Metrics 
from sklearn.metrics import confusion_matrix,accuracy_score
print("Training accuracy", accuracy_score(Y_train,Y_pred_train).round(3))
print("Testing accuracy", accuracy_score(Y_test,Y_pred_test).round(3))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train,Y_pred_train)
cm

#===================================================================
# validation set approch 
TrE = []
TsE = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    scores1 = accuracy_score(Y_train,Y_pred_train)
    scores2 = accuracy_score(Y_test,Y_pred_test)
    TrE.append(scores1.mean())
    TsE.append(scores2.mean())

print(TrE)
print(TsE)

#====================================================================
# K-fold cross-validation
from sklearn.model_selection import KFold,cross_val_score
kfold = KFold(n_splits=5)
KNN = KNeighborsClassifier(n_neighbors=5,p=1)
scores = cross_val_score(KNN, X, Y, cv=kfold)

print("cross validation sores: ",scores)
print("average cv score: ",scores.mean())
print('Number of CV scores used in Average: ',len(scores))










