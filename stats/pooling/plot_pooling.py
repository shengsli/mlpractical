import numpy as np
import matplotlib.pyplot as plt

run_times = 2
for pool in ['avg_pooling', 'max_pooling']:
    all_data = np.empty((run_times,100,4))
    all_test = np.empty((run_times,2))
    all_time = np.empty(run_times)
    all_best_epoch = np.empty((run_times,2))
    for run in range(0,run_times):
        all_data[run,:,:] = np.loadtxt(
            '{0}/{1}run/result_outputs/summary.csv'.format(pool,run+6),
            delimiter=',', skiprows=1)
        all_test[run,:] = np.loadtxt(
            '{0}/{1}run/result_outputs/test_summary.csv'.format(pool,run+6),
            delimiter=',', skiprows=1)
        # all_time[run] = np.loadtxt(
        #     '{0}/{1}run/result_outputs/time.csv'.format(pool,run+1),
        #     delimiter=',', skiprows=1).sum()
    avg_data = all_data.mean(axis=0)
    avg_test = all_test.mean(axis=0)
    std_test = all_test.std(axis=0)
    # avg_time = all_time.mean()
    # std_time = all_time.std()
    best_epoch = np.argmin(avg_data[:,3])
    # print("%d & %.3f$\pm$%.3f & %.3f$\pm$%.3f & %.1f$\pm$%.1f" %
    #       (best_epoch, 
    #        avg_test[0], std_test[0],
    #        avg_test[1], std_test[1],
    #        avg_time/60, std_time))
    print("%d & %.3f$\pm$%.3f & %.3f$\pm$%.3f" %
          (best_epoch, 
           avg_test[0], std_test[0],
           avg_test[1], std_test[1]))
    
    # plot
    plt.plot(avg_data[:,3], label=pool)

# display
plt.legend()
plt.title('Val loss of CNNs using pooling')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('pooling.pdf')
plt.show()
