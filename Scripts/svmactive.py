"""
Created on Mon Dec 15
@author: Sruthi,Ishita
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.lines as mlines
import pandas as pd
import numpy.ma as ma
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

# Global dictionary of index:data point mapping to be used by the oracle
mydictx, mydicty = {}, {}

c = 1
k = 'rbf'
# cparam = 1e+33
def processdata():
    # DATA PROCESSING
    # Loading data from CSV
    # Remove religion string as this information is already encoded in individual binary features
    # Remove patientID and INTNR as they are patient identifiers and not features
    df = pd.read_csv('WomenHealth_Training.csv', sep=',')
    data = df.drop(['religion', 'patientID', 'INTNR'], axis=1).values
    count = 0
    validdatalist = []
    # Count data with missing values (NaN) in one or many features
    # If all features are present, add it to the valid data list
    # Example: pd.isnull(pd.Series(['apple', np.nan, 'banana'])) = False True False
    for item in data:
        if True in pd.isnull(item):
            count = count + 1
        else:
            validdatalist.append(item)
    npvaliddata = np.asarray(validdatalist)
    print npvaliddata.shape
    X = npvaliddata[:len(npvaliddata), :npvaliddata.shape[1] - 1]
    Y = npvaliddata[:len(npvaliddata), npvaliddata.shape[1] - 1]
    print X.shape, Y.shape
    np.save('X.npy', X)
    np.save('Y.npy', Y)

def removelabels(n, Y):
    # Inputs n = int, number of labels to remove
    # Returns Y with n labels removed (removed labels have value 9999)
    if n <= len(Y):
        randomorder = np.random.choice(len(Y), n, replace=False)
        # print randomorder
        for index in randomorder:
            # 9999 indicates label has been removed
            Y[index] = 9999
    else:
        print "Number of labels to remove must be less than total number of available labels"
        return
    return Y


def partition(X, Y):
    # Inputs X = Training data cases, Y = Array with some labels removed
    # Returns UX (unlabeled X),UY(unlabeled Y), LX(labeled X), LY(labeled Y)
    UX, UY, LX, LY = [], [], [], []
    for i in range(0, len(X)):
        if Y[i] == 9999:
            UX.append(X[i])
            UY.append(Y[i])
        else:
            LX.append(X[i])
            LY.append(Y[i])
    return UX, UY, LX, LY

def SVMmodel(LX, LY, c, k):
    # Inputs LX (data points), LY (labels for data points in LX),C = inverse regularization strength
    # Returns parameter values
    clf = SVC(C=c,kernel=k).fit(LX, LY)
    return clf


def ActiveLearning(UX, UY, LX, LY, T):
    global c
    global k
    for t in range(0, T):
        svmobj = SVMmodel(LX, LY,c,k)
        # Entropy utility measure (ACTIVE LEARNING)
        queryindex = np.argmin(np.abs(svmobj.decision_function(UX)))
        # print queryindex,UX[queryindex]
        truelabel = oracle(UX[queryindex].tolist())
        # print truelabel
        LX = np.append(LX, [UX[queryindex]], axis=0)
        LY = np.append(LY, truelabel)
        UX = np.delete(UX, queryindex, axis=0)
        UY = np.delete(UY, queryindex, axis=0)
    # Train for the last time with all the labeled data
    svmobj = SVMmodel(LX, LY, c,k)
    return svmobj

def RandomSampling(UX, UY, LX, LY, T):
    global c
    global k
    print('unlabelled:' + str(len(UY)))
    print('labelled:' + str(len(LY)))
    for t in range(0, T):
        svmobj = SVMmodel(LX, LY, c,k)
        # Random sampling (PASSIVE LEARNING)
        queryindex = np.random.randint(0, len(UX))
        # print queryindex,UX[queryindex]
        truelabel = oracle(UX[queryindex].tolist())
        # print truelabel
        LX = np.append(LX, [UX[queryindex]], axis=0)
        LY = np.append(LY, truelabel)
        UX = np.delete(UX, queryindex, axis=0)
        UY = np.delete(UY, queryindex, axis=0)
    # Train for the last time with all the labeled data
    print('unlabelled:'+str(len(UY)))
    print('labelled:' + str(len(LY)))
    svmobj = SVMmodel(LX, LY, c,k)
    return svmobj

def oracle(X):
    # Input X, query for true label
    # Return true label
    global mydictx, mydicty
    dataindex = mydictx.keys()[mydictx.values().index(X)]
    return mydicty[dataindex]


def main():
    #processdata()
    # Load data
    X = np.load('X.npy')  # (1417L, 49L)
    Y = np.load('Y.npy')  # (1417L,)
    print X.shape
    #scale data
    # scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    # X = scaler.fit_transform(X)

    Xtr = X[:1000]
    Ytr = Y[:1000]
    Xte = X[1000:1417]
    Yte = Y[1000:1417]
    print Xtr.shape, Ytr.shape, Xte.shape, Yte.shape

    global mydictx, mydicty,c,k
    # Initialize mydictx and mydicty
    for i in range(0, len(Xtr)):
        mydictx[i] = Xtr[i].tolist()
        mydicty[i] = Ytr[i].tolist()
    clf = SVMmodel(Xtr, Ytr, c,k)
    print "Accuracy with SVM(C=1):", clf.score(Xte, Yte)

    #Let us remove 450 labels and start with 50 labelled points
    # yremoved = removelabels(450,Ytr[:500])
    # UX,UY,LX,LY = partition(Xtr[:500],yremoved)
    # np.save('UXtr450.npy', UX)
    # np.save('UYtr450.npy', UY)
    # np.save('LXtr50.npy', LX)
    # np.save('LYtr50.npy', LY)
    # UX = np.load('UXtr450.npy')
    # UY = np.load('UYtr450.npy')
    # LX = np.load('LXtr50.npy')
    # LY = np.load('LYtr50.npy')
    # UX = np.load('UXtr100.npy')
    # UY = np.load('UYtr100.npy')
    # LX = np.load('LXtr100.npy')
    # LY = np.load('LYtr100.npy')
    UX = np.load('UXtr1500.npy')
    UY = np.load('UYtr1500.npy')
    LX = np.load('LXtr1500.npy')
    LY = np.load('LYtr1500.npy')
    ALaccuracy,RSaccuracy = [],[]
    #Start with 50 and increase number of queries the learner can ask by 50 every iteration
    for tval in range(100,1000,100):
        print tval
        ALsvcobj = ActiveLearning(UX, UY, LX, LY, T=tval)
        ALaccuracy.append(ALsvcobj.score(Xte,Yte))

        RSsvcobj = RandomSampling(UX, UY, LX, LY, T=tval)
        RSaccuracy.append(RSsvcobj.score(Xte,Yte))
    print 'Active'
    print ALaccuracy
    print 'Random'
    print RSaccuracy
    np.save('ALSV_1500.npy',np.array(ALaccuracy))
    np.save('RSSV_1500.npy',np.array(RSaccuracy))

    ALaccuracy = np.load('ALSV_1500.npy')
    RSaccuracy = np.load('RSSV_1500.npy')
    fig, ax = plt.subplots()
    xaxis = np.arange(100,1000,100)
    ax.plot(xaxis, ALaccuracy, color='g', label='Active Learning')
    ax.plot(xaxis, RSaccuracy, color='r', label='Passive Learning')
    legend = ax.legend(loc=4, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.85')
    plt.xlabel('Number of Queries ---->')
    plt.ylabel('Accuracy---->')
    plt.title(
        r'Active Learning vs Passive Learning')
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    main()

