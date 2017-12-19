import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

#========================================================
AL_LR_map_1500 = np.load('ALLRmap_1500.npy')
print AL_LR_map_1500
AL_SV_1500 = np.load('ALSV_1500.npy')
RS_LR_map_1500 = np.load('RSLRmap_1500.npy')
print RS_LR_map_1500
RS_SV_1500 = np.load('RSSV_1500.npy')

xaxis = np.arange(10,100,10)

plt.title("SVC vs LR Active Learning")
plt.xlabel('No of Queries')
plt.ylabel('Accuracy')

p1, = plt.plot(xaxis, AL_LR_map_1500, 'b', label='Logistic Regression MAP Active Learning')
p2, = plt.plot(xaxis, AL_SV_1500, 'g', label='Support Vector Classification Active Learning')
p3, = plt.plot(xaxis, RS_LR_map_1500, '--b', label='Logistic Regression MAP RandomSampling')
p4, = plt.plot(xaxis, RS_SV_1500, '--g', label='Support Vector Classification RandomSampling')

plt.ylim(0.5,1.0)

plt.legend(handles=[p1, p2, p3, p4],
                   loc='lower right',
                   fontsize=10)
plt.savefig("../Results/SVC_LR_1500.pdf")
plt.show()

