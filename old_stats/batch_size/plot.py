import numpy as np
import matplotlib.pyplot as plt
import csv

# stats_dict = {
#     "train_acc": data[:,0],
#     "train_loss": data[:,1],
#     "val_acc": data[:,2],
#     "val_loss": data[:,3]
# }

for batch_size in [20,40,200]:
    best_model_data = np.loadtxt('max_pooling_{0}batch_size/result_outputs/test_summary.csv'.format(batch_size),
                      delimiter=',', skiprows=1)
    data = np.loadtxt('max_pooling_{0}batch_size/result_outputs/summary.csv'.format(batch_size),
                      delimiter=',', skiprows=1)
    best_val_epoch = np.argmin(data[:,3])
    print("%d & %.3f & %.3f" % (best_val_epoch,best_model_data[0],best_model_data[1]))
    plt.plot(data[:,3], label='{0} batch size'.format(batch_size))

plt.legend()
plt.title('')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
