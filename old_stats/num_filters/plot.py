import numpy as np
import matplotlib.pyplot as plt
import csv

# stats_dict = {
#     "train_acc": data[:,0],
#     "train_loss": data[:,1],
#     "val_acc": data[:,2],
#     "val_loss": data[:,3]
# }

for num_filters in [4,16,32]:
    best_model_data = np.loadtxt('max_pooling_4layers_{0}filters/result_outputs/test_summary.csv'.format(num_filters),
                      delimiter=',', skiprows=1)
    data = np.loadtxt('max_pooling_4layers_{0}filters/result_outputs/summary.csv'.format(num_filters),
                      delimiter=',', skiprows=1)
    best_val_epoch = np.argmin(data[:,3])
    print("%d & %.3f & %.3f" % (best_val_epoch,best_model_data[0],best_model_data[1]))
    plt.plot(data[:,3], label='{0}filters'.format(num_filters))

plt.legend()
plt.title('')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
