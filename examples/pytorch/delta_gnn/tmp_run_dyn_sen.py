import os
import datetime
import time
import util
import subprocess

# deg_th_list_small_dataset = [1, 2, 5, 10, 20, 30]
# deg_th_list_large_dataset = [50, 100, 300, 500, 1000, 2000]
# deg_th_list_large_dataset = [10, 20, 30, 50, 100]
# deg_th_list_large_dataset = [10, 20]
# threshold_list = [1]
threshold_list = [2, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 25, 30, 50]
# threshold_list = [5, 10, 13, 15, 18, 20, 23, 25, 28, 30, 35, 40, 50]
model_list = ['gcn']
# model_list = ['gat']
# dataset_list = ['Cora', 'CiteSeer']
# dataset_list = ['arxiv']
# dataset_list = ['reddit']
# dataset_list = ['products']
dataset_list = ['mag']
# dataset_list = ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-mag']
# dataset_list = ['ogbn-arxiv', 'ogbn-mag']
# epoch = 200
# task_num = len(deg_th_list)

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

# Predefined
log_path = './results/log/' + theTime + '_log.txt'

# Remove previous log
os.system('rm ' + log_path)

for dataset in dataset_list:
    for model in model_list:
        # os.system('pwd')
        for deg_th in threshold_list:
            os.system('python3.8 ./tmp_train_dyn_sen.py' + ' --dataset=' + dataset + ' --model=' +
                      model + ' --threshold=' + str(deg_th))

print('\n>> All Tasks finish, total execution time: {:.4}s'.format(
    util.time_format(time.perf_counter() - Task_time_start)))
