import os
import datetime
import time

deg_interval_cora = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                     [9, 11], [11, 16], [16, 168]]
deg_interval_citeseer = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                         [9, 11], [11, 16], [16, 99]]
deg_interval_actor = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                      [9, 11], [11, 16], [16, 73]]
# deg_interval_facebook = [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [15, 18], [18, 27], [27, 36],
#                          [36, 54], [54, 72], [72, 90], [90, 710]]
deg_interval_facebook = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 20],
                         [20, 26], [26, 32], [32, 50], [50, 710]]
deg_interval_wikics = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70],
                       [70, 80], [80, 90], [90, 110], [110, 160], [160, 3324]]

# dataset_list = ['Cora', 'CiteSeer']
dataset_list = ['FacebookPagePage', 'WikiCS']
# dataset_list = ['WikiCS']
# dataset_list = ['FacebookPagePage']
degree_type_list = ['outdeg', 'indeg']
# degree_type_list = ['outdeg']

# noise_intensity = [0.1, 0.5, 1, 2, 5]
# noise_intensity = [0.1, 0.5, 1]
# noise_intensity = [1, 2, 5]
# noise_intensity = [8, 10]
# node_ratio = [0.1, 0.2, 0.5, 1]
# node_ratio = [0.05]
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
    elif dataset == 'Actor':
        deg_interval = deg_interval_actor
    elif dataset == 'WikiCS':
        deg_interval = deg_interval_wikics
    elif dataset == 'FacebookPagePage':
        deg_interval = deg_interval_facebook

    if dataset == 'Cora' or dataset == 'CiteSeer' or dataset == 'WikiCS':
        node_ratio = [0.02]
        noise_intensity = [1, 2, 5]
    elif dataset == 'FacebookPagePage':
        node_ratio = [0.05]
        noise_intensity = [5, 8, 10]

    for intensity in noise_intensity:
        for degree_type in degree_type_list:
            for deg_region in deg_interval:
                deg_begin = deg_region[0]
                deg_end = deg_region[1]
                for ratio in node_ratio:
                    # Task batch
                    print(dataset)
                    print("\n*** Deg: [{:d}, {:d}), Node_ratio: {:.2f}, Intensity: {:.2f}".format(
                        deg_begin, deg_end, ratio, intensity))

                    # deg_begin = str(deg_begin)
                    # deg_end = str(deg_end)
                    # intensity = str(intensity)
                    # ratio = str(node_ratio)

                    log_path = './results/sensitivity/' + dataset + '_' + degree_type + '_intense_' + str(
                        intensity) + '_deg_' + str(deg_begin) + '-' + str(deg_end) + '.txt'
                    os.system('rm ' + log_path)

                    os.system('/usr/bin/python3.8 ./gcn_sensitive.py --dataset=' + dataset +
                              ' --n-epochs=' + epoch_str + ' --degree-type=' + degree_type +
                              ' --deg-begin=' + str(deg_begin) + ' --deg-end=' + str(deg_end) +
                              ' --noise-intensity=' + str(intensity) + ' --node-ratio=' +
                              str(ratio) + ' | tee -a ' + log_path)

print('\n>> Task execution time: {:.4}s'.format(time.perf_counter() - Task_time_start))