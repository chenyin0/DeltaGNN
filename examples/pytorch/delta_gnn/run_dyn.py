import os
import datetime
import time
import util
import subprocess

# deg_th_list_small_dataset = [1, 2, 5, 10, 20, 30]
# deg_th_list_large_dataset = [50, 100, 300, 500, 1000, 2000]
# deg_th_list_large_dataset = [10, 20, 30, 50, 100]
# deg_th_list_large_dataset = [10, 20]
threshold_list = [1, 2, 5, 10, 20, 30, 50]
model_list = ['gcn', 'graphsage', 'gat', 'gin']
# dataset_list = ['Cora', 'CiteSeer', 'PubMed', 'arxiv']
dataset_list = ['arxiv']
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

# os.system('cd ./examples/pytorch/delta_gnn/')

# for dataset in dataset_list:
#     for arch in arch_list:
#         # os.system('pwd')
#         if dataset == 'cora' or dataset == 'citeseer':
#             epoch = 200
#             os.system('/usr/bin/python3.8 ./train_evo.py' + ' --dataset=' + dataset + ' --model=' +
#                       arch + ' --n-epochs=' + str(epoch))

#             for deg_th in deg_th_list:
#                 os.system('/usr/bin/python3.8 ./train_evo_delta.py' + ' --dataset=' + dataset +
#                           ' --model=' + arch + ' --deg-threshold=' + str(deg_th) + ' --n-epochs=' +
#                           str(epoch))
#         elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
#             epoch = 100
#             os.system('/usr/bin/python3.8 ./train_evo.py' + ' --dataset=' + dataset + ' --model=' +
#                       arch + ' --n-epochs=' + str(epoch))
#             for deg_th in deg_th_list_large_dataset:
#                 os.system('/usr/bin/python3.8 ./train_evo_delta.py' + ' --dataset=' + dataset +
#                           ' --model=' + arch + ' --deg-threshold=' + str(deg_th) + ' --n-epochs=' +
#                           str(epoch))

for dataset in dataset_list:
    for model in model_list:
        # os.system('pwd')
        for deg_th in threshold_list:
            os.system('python3.8 ./train_dyn.py' + ' --dataset=' + dataset + ' --model=' + model +
                      ' --threshold=' + str(deg_th))

print('\n>> All Tasks finish, total execution time: {:.4}s'.format(
    util.time_format(time.perf_counter() - Task_time_start)))
