import os
import datetime
import time

deg_interval_cora = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                     [9, 11], [11, 16], [16, 168]]
deg_interval_citeseer = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                         [9, 11], [11, 16], [16, 99]]

dataset_list = ['Cora', 'CiteSeer']
degree_type_list = ['outdeg', 'indeg']

# noise_intensity = [0.1, 0.5, 1, 2, 5]
# noise_intensity = [0.1, 0.5, 1]
noise_intensity = [1, 2, 5]
# node_ratio = [0.1, 0.2, 0.5, 1]
node_ratio = [0.02]
epoch = 200
# task_num = len(deg_interval) * len(noise_intensity)

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

# Predefined
# log_path = './results/sensitivity/' + theTime + '_log.txt'

# Remove previous log
# os.system('rm ' + log_path)

epoch_str = str(epoch)

for dataset in dataset_list:
    if dataset == 'Cora':
        deg_interval = deg_interval_cora
    elif dataset == 'CiteSeer':
        deg_interval = deg_interval_citeseer

    for degree_type in degree_type_list:
        for deg_region in deg_interval:
            deg_begin = deg_region[0]
            deg_end = deg_region[1]
            for intensity in noise_intensity:
                for ratio in node_ratio:
                    # Task batch
                    print(dataset)
                    print("\n*** Deg: [{:d}, {:d}), Node_ratio: {:.2f}, Intensity: {:.2f}".format(
                        deg_begin, deg_end, ratio, intensity))

                    # deg_begin = str(deg_begin)
                    # deg_end = str(deg_end)
                    # intensity = str(intensity)
                    # ratio = str(node_ratio)

                    log_path = './results/sensitivity/' + dataset + '_' + degree_type + '_intense_' + str(intensity) + '_deg_' + str(deg_begin) + '-' + str(deg_end) + '.txt'
                    os.system('rm ' + log_path)

                    os.system('/usr/bin/python3.8 ./gcn_sensitive.py --dataset=' + dataset + ' --n-epochs=' +
                            epoch_str + ' --degree-type='+ degree_type + ' --deg-begin=' + str(deg_begin) + ' --deg-end=' +
                            str(deg_end) + ' --noise-intensity=' + str(intensity) + ' --node-ratio=' +
                            str(ratio) + ' | tee -a ' + log_path)

print('\n>> Task execution time: {:.4}s'.format(time.perf_counter() - Task_time_start))