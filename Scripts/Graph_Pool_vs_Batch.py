import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

#===========================Time=============================
pool_time = np.load('pooltime.npy')[:4]
batch_time = np.load('batchtime.npy')[:4]
xaxis = np.arange(10,50,10)

plt.title("Bayesian Inference Pooling vs Batching Time")
plt.xlabel('No of Queries')
plt.ylabel('Time')

p1, = plt.plot(xaxis, pool_time, 'b', label='Pooling Time')
p2, = plt.plot(xaxis, batch_time, 'green', label='Batching Time')

#plt.ylim(0.5,1)

plt.legend(handles=[p1, p2],
                   loc='lower right',
                   fontsize=10)
plt.savefig("../Results/Pool_vs_Batch_time_50.pdf")
plt.show()

#===========================Accuracy=============================
batch_accuracy = np.load('BALAccu_S100.npy')
pool_accuracy = np.load('PoolBay.npy')
xaxis = np.arange(10,100,10)

plt.title("Bayesian Inference Pooling vs Batching Accuracy")
plt.xlabel('No of Queries')
plt.ylabel('Accuracy')

p1, = plt.plot(xaxis, pool_accuracy, 'b', label='Pooling Accuracy')
p2, = plt.plot(xaxis, batch_accuracy, 'green', label='Batching Accuracy')

plt.ylim(0.4,1)

plt.legend(handles=[p1, p2],
                   loc='lower right',
                   fontsize=10)
plt.savefig("../Results/Pool_vs_Batch_Accuracy.pdf")
plt.show()
