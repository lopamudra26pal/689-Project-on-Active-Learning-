# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:43:55 2017

@author: Sruthi, Lopa
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.special import logsumexp
import matplotlib.lines as mlines
import pandas as pd
import numpy.ma as ma
from sklearn.preprocessing import MinMaxScaler
import sys
import math
np.random.seed(0)

#Global dictionary of index:data point mapping to be used by the oracle
mydictx,mydicty = {},{}
cvalue = 1e+33
svalue = 100
batchsize = 5

def processdata():
    #DATA PROCESSING
    #Loading data from CSV 
    df=pd.read_csv('WomenHealth_Training.csv', sep=',')
    #Remove religion string as this information is already encoded in individual 
    #religion binary features
    #Remove patientID and INTNR as they are patient identifiers and not features
    data = df.drop(['religion','patientID','INTNR'], axis=1).values
    #print data.shape
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
    #print npvaliddata.shape
    X = npvaliddata[:len(npvaliddata),:npvaliddata.shape[1]-1]
    Y = npvaliddata[:len(npvaliddata),npvaliddata.shape[1]-1]
    #print count
    #print X.shape, Y.shape
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
    #Returns lists UX (unlabeled X),UY(unlabeled Y), LX(labeled X), LY(labeled Y)
    UX,UY,LX,LY = [],[],[],[]
    for i in range(0,len(X)):
        if Y[i] == 9999:
            UX.append(X[i])
            UY.append(Y[i])
        else:
            LX.append(X[i])
            LY.append(Y[i])
    return UX,UY,LX,LY

def rejection_sampler(S,X,Y,mu,sigma):
    #Inputs S = number of samples (int), X,Y = training data (numpy arrays)
    #Mu = mean vector (array), sigma = covariance matrix (2D array)
    #Output = array of S samples from P(theta|D) parameter posterior
    thetars = []
    logregobj = LogisticRegression(C=cvalue)
    returnedobj = logregobj.fit(X,Y)
    wmle = np.asarray(returnedobj.coef_[0])
    bmle = np.asarray(returnedobj.intercept_[0])
    pmlesum = 0
    #MLE computation has numerical stability issues, look at the final script version for stable computation
    for i in range(len(X)):
        expcoeff = np.exp(-(wmle.dot(X[i].T)+bmle))
        if Y[i] == 1:
            pmlesum = pmlesum - np.log(1 + expcoeff)
        else:
            pmlesum = pmlesum + np.log(expcoeff/(1 + expcoeff))
    #pmle = np.exp(pmlesum)
    #print 'Rejection Sampler pmlesum: ',pmlesum
    for s in range(S):
        print 'Sample size',s,'out of ',S
        accept = False
        while(accept == False):
            thetas = np.random.multivariate_normal(mu,sigma)
            ws = thetas[:len(thetas)-1]
            bs = thetas[len(thetas)-1]
            u = np.random.uniform()
            pssum = 0
            for i in range(len(X)):
                #print np.exp(-(ws.dot(X[i].T)+bs))
                #expcoeff = np.exp(-(ws.dot(X[i].T)+bs))
                if Y[i] == 1:
                    pssum = pssum - logsumexp([0,-(ws.dot(X[i].T)+bs)])
                else:
                    pssum = pssum - logsumexp([0,(ws.dot(X[i].T)+bs)])
            #ps = np.exp(pssum)
            #if (ps/pmle >= u):
            if pssum - pmlesum >= math.log(u):
                accept = True
                thetars.append(thetas)  
    return np.array(thetars)

def predictive_distribution(S,X,theta):
    #Inputs S = Number of samples in theta (int)
    #X = single data point for which label must be predicted
    #Theta = Set of theta values (array), each theta is a vector containing [w1,w2..wn,b]
    #Output = P(Y=y|X=x,D,mu,sigma) (Posterior predictive distribution)
    psum0,psum1 = 0,0
    numfeatures = theta.shape[1]
    for s in range(S):
        w = np.array(theta[s][:numfeatures-1])
        b = theta[s][numfeatures-1]
        expcoeff = np.exp(-(w.dot(X.T)+b))
        psum1 = psum1 + (1/(1 + expcoeff))
        psum0 = psum0 + (expcoeff/(1 + expcoeff))
    return psum0/S,psum1/S

def BayesianPredict(S,UX,theta):
    #Inputs S = number of samples in theta
    #UX = Unlabeled X for which label probabilities must be predicted ()
    #Theta = Set of theta values (array), each theta is a vector containing [w1,w2..wn,b]
    #Returns predicted probability of labels for every data point in UX
    predictedy = []
    #print len(UX[0])
    for i in range(0,len(UX)):
        psum0,psum1 = predictive_distribution(S,UX[i],theta)
        predictedy.append(np.array([psum0,psum1]))
    return np.array(predictedy)

def LogRegTrain(LX,LY,cparam):
    #Inputs LX (data points), LY (labels for data points in LX),C = inverse regularization strength
    #Returns parameter values
    logregobj = LogisticRegression(C=cparam,random_state=0)
    returnedobj = logregobj.fit(LX,LY)
    wmle = np.asarray(returnedobj.coef_[0])
    bmle = np.asarray(returnedobj.intercept_[0])
    return wmle,bmle,returnedobj

def BayesianActiveLearning(UX,UY,LX,LY,T):
    global cvalue,svalue,batchsize
    #Zero mean prior vector of length of number of features + 1 for bias
    mu = np.zeros(np.array(LX).shape[1]+1) 
    #Variance covariance matrix of 100I 
    sigma = 100*np.identity(np.array(LX).shape[1]+1)
    for t in range(0,T):
        #Drawing 100 samples from parameter posterior
        thetars = rejection_sampler(svalue,LX,LY,mu,sigma)
        predictedy = BayesianPredict(svalue,UX,thetars)
        #Using batching to reduce the number of times the model is trained
        for i in range(batchsize):
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
            predictedy = np.delete(predictedy,queryindex,axis=0) 
    #Train for the last time with all the labeled data
    thetars = rejection_sampler(svalue,LX,LY,mu,sigma)
    return thetars

def BayesianRandomSampling(UX,UY,LX,LY,T):
    global cvalue,svalue,batchsize
    #Zero mean prior vector of length of number of features + 1 for bias
    mu = np.zeros(np.array(LX).shape[1]+1)
    #Variance covariance matrix of 100I
    sigma = 100*np.identity(np.array(LX).shape[1]+1)
    for t in range(0,T):
        #Drawing 100 samples from parameter posterior
        thetars = rejection_sampler(svalue,LX,LY,mu,sigma)
        predictedy = BayesianPredict(svalue,UX,thetars)
        #Using batching to reduce the number of times the model is trained
        for i in range(batchsize):
            #Entropy utility measure (ACTIVE LEARNING)
            queryindex = np.random.randint(0, len(UX))
            truelabel = oracle(UX[queryindex].tolist())
            LX = np.append(LX,[UX[queryindex]],axis=0)
            LY = np.append(LY,truelabel)
            UX = np.delete(UX,queryindex,axis=0)
            UY = np.delete(UY,queryindex,axis=0)
            predictedy = np.delete(predictedy,queryindex,axis=0)
    #Train for the last time with all the labeled data
    thetars = rejection_sampler(svalue,LX,LY,mu,sigma)
    return thetars



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
    #scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    #X = scaler.fit_transform(X)
    Xtr = X[:1000]
    Ytr = Y[:1000]
    # Xte = X[1000:1417]
    # Yte = Y[1000:1417]
    Xte = X[1000:1100]
    Yte = Y[1000:1100]
    print 'Xtr.shape',Xtr.shape
    print 'Ytr.shape',Ytr.shape
    print 'Xte.shape',Xte.shape
    print 'Yte.shape',Yte.shape
 
    global mydictx,mydicty,cvalue,svalue,batchsize
    #Initialize mydictx and mydicty
    for i in range(0,len(Xtr)):
        mydictx[i] = Xtr[i].tolist()
        mydicty[i] = Ytr[i].tolist()
    #C=1000 gives 81.5 over whole data
    #for m in range(0,)
    print 'cvalue',cvalue
    logregobj = LogisticRegression(C=cvalue,random_state=0)
    #returnedobj = logregobj.fit(Xtr[:50],Ytr[:50])
    returnedobj = logregobj.fit(Xtr[50:100], Ytr[50:100])
    #ytrlr = returnedobj.predict(Xtr)
    #print accuracy_score(Ytr,ytrlr)
    ytelr = returnedobj.predict(Xte)
    print 'ytelr.shape',ytelr.shape
    print 'Accuracy score for model trained on 50 data-cases: ',accuracy_score(Yte,ytelr)
    print '***********************'
    
    #Let us remove 450 labels and start with 50 labelled points
    # yremoved = removelabels(45,Ytr[:50])
    # UX,UY,LX,LY = partition(Xtr[:50],yremoved)
    yremoved = removelabels(20, Ytr[:100])
    UX, UY, LX, LY = partition(Xtr[:100], yremoved)
    print 'UX.shape',len(UX)
    print 'LX.shape', len(LX)
    print 'UY.shape', len(UY)
    print 'LY.shape', len(LY)
    RSaccuracy,BALaccuracy = [],[]
    #Start with 50 and increase number of queries the learner can ask by 50 every iteration
    for tval in range(20,100,20):
        print 'tval',tval
        theta_rs = BayesianRandomSampling(UX, UY, LX, LY, T=int(tval / batchsize))
        np.save('thetars.npy', theta_rs)
        brsyte = np.argmax(BayesianPredict(svalue, Xte, theta_rs), axis=1)
        print 'accuracy_score_rs', accuracy_score(Yte, brsyte)
        RSaccuracy.append(accuracy_score(Yte, brsyte))

        thetars = BayesianActiveLearning(UX,UY,LX,LY,T=int(tval/batchsize))
        np.save('thetars.npy',thetars)
        balyte = np.argmax(BayesianPredict(svalue,Xte,thetars),axis=1)
        print 'accuracy_score_al',accuracy_score(Yte,balyte)
        BALaccuracy.append(accuracy_score(Yte,balyte))
    print RSaccuracy,BALaccuracy
    np.save('BALAccu100_1.npy', np.array(BALaccuracy))
    np.save('BRSAccu100_1.npy', np.array(RSaccuracy))


if __name__ == "__main__":
  main()
