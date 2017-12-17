# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:40:38 2017

@author: Sruthi
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.stats import entropy
import matplotlib.lines as mlines
import pandas as pd
import numpy.ma as ma
np.random.seed(0)

#Global dictionary of index:data point mapping to be used by the oracle
mydictx,mydicty = {},{}
cvalue = 1e+33

def processdata():
    #DATA PROCESSING
    #Loading data from CSV 
    df=pd.read_csv('WomenHealth_Training.csv', sep=',')
    #Remove religion string as this information is already encoded in individual 
    #religion binary features
    #Remove patientID and INTNR as they are patient identifiers and not features
    data = df.drop(['religion','patientID','INTNR'], axis=1).values
    print data.shape
    count = 0
    validdatalist = []
    #Count data with missing values (NaN) in one or many features
    #If all features are present, add it to the valid data list
    #Example: pd.isnull(pd.Series(['apple', np.nan, 'banana'])) = False True False 
    for item in data:
        if True in pd.isnull(item):
            count = count + 1
        else:
            validdatalist.append(item)
    npvaliddata = np.asarray(validdatalist)
    print npvaliddata.shape
    X = npvaliddata[:len(npvaliddata),:npvaliddata.shape[1]-1]
    Y = npvaliddata[:len(npvaliddata),npvaliddata.shape[1]-1]
    print count
    print X.shape, Y.shape
    #Encode labels (1,2) to (0,1)
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    #print Y
    np.save('X.npy',X)
    np.save('Y.npy',Y)

def removelabels(n,Y):
    #Inputs n = int, number of labels to remove
    #Returns Y with n labels removed (removed labels have value 9999)
    if n <= len(Y):
        randomorder = np.random.choice(len(Y),n,replace=False)
        #print randomorder
        for index in randomorder:
            #9999 indicates label has been removed
            Y[index] = 9999
    else:
        print "Number of labels to remove must be less than total number of available labels"
        return
    return Y

def partition(X,Y):
    #Inputs X = Training data cases, Y = Array with some labels removed
    #Returns UX (unlabeled X),UY(unlabeled Y), LX(labeled X), LY(labeled Y)
    UX,UY,LX,LY = [],[],[],[]
    for i in range(0,len(X)):
        if Y[i] == 9999:
            UX.append(X[i])
            UY.append(Y[i])
        else:
            LX.append(X[i])
            LY.append(Y[i])
    return UX,UY,LX,LY

def LogRegTrain(LX,LY,cparam):
    #Inputs LX (data points), LY (labels for data points in LX),C = inverse regularization strength
    #Returns parameter values
    logregobj = LogisticRegression(C=cparam,random_state=0)
    returnedobj = logregobj.fit(LX,LY)
    wmle = np.asarray(returnedobj.coef_[0])
    bmle = np.asarray(returnedobj.intercept_[0])
    return wmle,bmle,returnedobj

def ActiveLearning(UX,UY,LX,LY,T):
    global cvalue
    for t in range(0,T):
        wmle,bmle,returnedobj = LogRegTrain(LX,LY,cparam=cvalue)
        predictedy = returnedobj.predict_proba(UX)
        #Entropy utility measure (ACTIVE LEARNING) 
        queryindex = np.argmax(np.sum(-predictedy*np.log(predictedy+1e-45),axis=1))
        #queryindex = np.argmax(entropy(predictedy.T).T)
        #print queryindex,predictedy.shape
        truelabel = oracle(UX[queryindex].tolist())
        #print truelabel       
        LX = np.append(LX,[UX[queryindex]],axis=0)
        LY = np.append(LY,truelabel)
        UX = np.delete(UX,queryindex,axis=0)
        UY = np.delete(UY,queryindex,axis=0)
    #Train for the last time with all the labeled data
    wmle,bmle,returnedobj = LogRegTrain(LX,LY,cparam=cvalue)
    return wmle,bmle,returnedobj

def RandomSampling(UX,UY,LX,LY,T):
    global cvalue
    for t in range(0,T):
        wmle,bmle,returnedobj = LogRegTrain(LX,LY,cparam=cvalue)
        #Random sampling (PASSIVE LEARNING)
        queryindex = np.random.randint(0,len(UX))
        #print queryindex,UX[queryindex]
        truelabel = oracle(UX[queryindex].tolist())
        #print truelabel
        LX = np.append(LX,[UX[queryindex]],axis=0)
        LY = np.append(LY,truelabel)
        UX = np.delete(UX,queryindex,axis=0)
        UY = np.delete(UY,queryindex,axis=0)
    #Train for the last time with all the labeled data
    wmle,bmle,returnedobj = LogRegTrain(LX,LY,cparam=cvalue)
    return wmle,bmle,returnedobj

def oracle(X): 
    #Input X, query for true label 
    #Return true label
    global mydictx,mydicty
    dataindex = mydictx.keys()[mydictx.values().index(X)]  
    return mydicty[dataindex]         
   
def main():
    #processdata()
    #Load data
    X = np.load('X.npy') #(1417L, 49L)
    Y = np.load('Y.npy') #(1417L,)
    Xtr = X[:1000]
    Ytr = Y[:1000]
    Xte = X[1000:1417]
    Yte = Y[1000:1417]
    print Xtr.shape,Ytr.shape,Xte.shape,Yte.shape
 
    global mydictx,mydicty,cvalue
    #Initialize mydictx and mydicty
    for i in range(0,len(Xtr)):
        mydictx[i] = Xtr[i].tolist()
        mydicty[i] = Ytr[i].tolist()
    #C=1000 gives 81.5 over whole data
    #for m in range(0,)
    logregobj = LogisticRegression(C=cvalue,random_state=0)
    returnedobj = logregobj.fit(Xtr[:500],Ytr[:500])
    #ytrlr = returnedobj.predict(Xtr)
    #print accuracy_score(Ytr,ytrlr)
    ytelr = returnedobj.predict(Xte)
    print accuracy_score(Yte,ytelr)
    
    """
    #Let us remove 450 labels and start with 50 labelled points
    yremoved = removelabels(450,Ytr[:500])
    UX,UY,LX,LY = partition(Xtr[:500],yremoved)
    ALaccuracy,RSaccuracy = [],[]
    #Start with 50 and increase number of queries the learner can ask by 50 every iteration
    for tval in range(50,500,50):
        
        wmle,bmle,ALreturnedobj = ActiveLearning(UX,UY,LX,LY,T=tval) 
        alyte = ALreturnedobj.predict(Xte)
        print tval,accuracy_score(Yte,alyte)
        ALaccuracy.append(accuracy_score(Yte,alyte))
        
        wmle,bmle,RSreturnedobj = RandomSampling(UX,UY,LX,LY,T=tval) 
        rsyte = RSreturnedobj.predict(Xte)
        print accuracy_score(Yte,rsyte)
        RSaccuracy.append(accuracy_score(Yte,rsyte))
    
    np.save('ALAccuDemo.npy',np.array(ALaccuracy))
    np.save('RSAccuDemo.npy',np.array(RSaccuracy))
    """
    
    #Plot
    ALaccuracy = np.load('ALAccuDemo.npy')
    RSaccuracy = np.load('RSAccuDemo.npy')
    #print ALaccuracy,RSaccuracy
    plt.title("Active Learning vs Passive Learning")
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy')
    xaxis = np.arange(50,500,50)
    plt.plot(xaxis,ALaccuracy,color='g',label='Active Learning')
    plt.plot(xaxis,RSaccuracy,color='r',label = 'Passive Learning')
    plt.legend()
    plt.show()     
     
       
if __name__ == "__main__":
  main()