import numpy as np
import matplotlib.pyplot as plt

run_times = 3
for stride in range(1,5):
    all_data = np.empty((run_times,100,4))
    all_time = np.empty(run_times)
    all_best_epoch = np.empty((run_times,2))
    for run in range(0,run_times):
        all_data[run,:,:] = np.loadtxt(
            '{0}stride/{1}run/result_outputs/summary.csv'.format(stride,run+1),
            delimiter=',', skiprows=1)
        all_time[run] = np.loadtxt(
            '{0}stride/{1}run/result_outputs/time.csv'.format(stride,run+1),
            delimiter=',', skiprows=1).sum()
    avg_data = all_data.mean(axis=0)
    avg_time = all_time.mean()
    print("%d stride, %.2f" % (stride, avg_time))

    # plot
    plt.plot(avg_data[:,3], label='{0}stride'.format(stride))

# display
plt.legend()
plt.title('Val loss of strided CNN')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
