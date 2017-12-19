import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

AL_C1e12 = np.load('SVC_AL_C=1e12.npy')
print AL_C1e12
RS_C1e12 = np.load('SVC_RS_C=1e12.npy')
AL_C1 = np.load('SVC_AL_C=1.npy')
print AL_C1
RS_C1 = np.load('SVC_RS_C=1.npy')
AL_C100 = np.load('SVC_AL_C=100.npy')
print AL_C100
RS_C100 = np.load('SVC_RS_C=100.npy')

plt.title("Performance of varying Regularization on Support Vector Classification")
plt.xlabel('Number of Queries')
plt.ylabel('Accuracy')
#xaxis = np.arange(10, 21, 5)
xaxis = np.arange(100,1000,100)
p1, = plt.plot(xaxis, AL_C1e12, 'g', label='Active Learning C=1e12')
p2, = plt.plot(xaxis, RS_C1e12, '--g', label='Random Sampling C=1e12')

p3, = plt.plot(xaxis, AL_C1, 'b', label='Active Learning C=1')
p4, = plt.plot(xaxis, RS_C1, '--b', label='Random Sampling C=1')

p5, = plt.plot(xaxis, AL_C100, 'r', label='Active Learning C=100')
p6, = plt.plot(xaxis, RS_C100, '--r',  label='Random Sampling C=100')

plt.ylim(0.5,1)

plt.legend(handles=[p1, p2, p3, p4, p5, p6],
                   loc='lower right',
                   fontsize=10)
plt.savefig("../Results/Compare_SVC_Diff_Reg.pdf")
plt.show()