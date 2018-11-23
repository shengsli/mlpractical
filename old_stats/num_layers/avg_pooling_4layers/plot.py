import numpy as np
import matplotlib.pyplot as plt
import csv

# stats_dict = {
#     "train_acc": data[:,0],
#     "train_loss": data[:,1],
#     "val_acc": data[:,2],
#     "val_loss": data[:,3]
# }

dim_reduction = "avg_pooling"
fig_title = "Val loss of max pooling"
save_fig_name = '{0}.pdf'.format(dim_reduction)

run_times = 2
all_data = np.empty((run_times,100,4))
all_total_time = np.empty((run_times))
all_best_model_data = np.empty((run_times,2))
all_best_val_epoch = np.empty(run_times)
for run in range(1,3):
    all_data[run-1,:,:] = np.loadtxt(
        '{0}run/result_outputs/summary.csv'.format(run),delimiter=',', skiprows=1)
    all_best_model_data[run-1,:] = np.loadtxt(
        '{0}run/result_outputs/test_summary.csv'.format(run), delimiter=',', skiprows=1)
avg_data = all_data.mean(axis=0)
avg_best_val_epoch = np.argmin(all_data[:,:,3], axis=1).mean()
avg_best_model_data = all_best_model_data.mean(axis=0)
print(avg_best_val_epoch)
print(avg_best_model_data)
plt.plot(avg_data[:,3], label='avg pooling')

plt.legend()
plt.title(fig_title)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(save_fig_name)
plt.show()
