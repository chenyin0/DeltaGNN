import os
import datetime

deg_interval = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 11],
                [11, 16], [16, 168]]
noise_intensity = [0.1, 0.5, 1, 2, 5]
node_ratio = [0.1, 0.2, 0.5, 1]
epoch = 200
# task_num = len(deg_interval) * len(noise_intensity)

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

# Predefined
log_path = './results/' + theTime + '_log.txt'

# Remove previous log
os.system('rm ' + log_path)

epoch_str = str(epoch)

for deg_region in deg_interval:
    deg_begin = deg_region[0]
    deg_end = deg_region[1]
    for intensity in noise_intensity:
        for ratio in node_ratio:
            # Task batch
            print("\n*** Deg: [{:d}, {:d}), Node_ratio: {:.2f}, Intensity: {:.2f}".format(
                deg_begin, deg_end, ratio, intensity))

            # deg_begin = str(deg_begin)
            # deg_end = str(deg_end)
            # intensity = str(intensity)
            # ratio = str(node_ratio)

            os.system('python ./examples/pytorch/gcn/gcn_sensitive.py --dataset=cora --n-epochs=' +
                      epoch_str + ' --deg-begin=' + str(deg_begin) + ' --deg-end=' + str(deg_end) +
                      ' --noise-intensity=' + str(intensity) + ' --node-ratio=' + str(ratio) +
                      ' | tee -a ' + log_path)
