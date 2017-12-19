import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

xaxis = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
yaxis = [0.74, 0.71, 0.70, 0.73, 0.69, 0.69, 0.72, 0.73, 0.73, 0.73]
#===============Taking the 5th indexed values only===================

plt.title("Effect of Unlabeled Sample Size on Bayesian AL")
plt.xlabel('Unlabeled Sampled Size')
plt.ylabel('Accuracy')

plt.ylim(0.65, 0.8)
p1, = plt.plot(xaxis, yaxis, 'b', marker='*', linewidth = 3)

plt.legend(title = "Data Sample Size = 50",
                   loc='lower right',
                   fontsize=10)
plt.savefig("../Results/Diff_Unlabeled_Sample_Size.pdf")
plt.show()