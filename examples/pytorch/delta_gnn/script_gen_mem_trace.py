import os
import datetime
import time
import util

deg_th_list_small_dataset = [1, 2, 5, 10, 20, 30]
deg_th_list_large_dataset = [10, 20, 50, 100, 300, 500]
# epoch = 200
# task_num = len(deg_th_list)
arch_list = ['hygcn', 'awb-gcn', 'i-gcn', 'regnn', 'delta-gnn', 'delta-gnn-opt']
# arch_list = ['delta-gnn', 'delta-gnn-opt']
dataset_list = ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-mag']

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

# Predefined
log_path = '../../../results/' + theTime + '_log.txt'

# Remove previous log
os.system('rm ' + log_path)

# for dataset in dataset_list:
#     for arch in arch_list:
#         if arch == 'delta-gnn' or arch == 'delta-gnn-opy':
#             if dataset == 'cora' or dataset == 'citeseer':
#                 deg_th_list = deg_th_list_small_dataset
#             elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
#                 deg_th_list = deg_th_list_large_dataset
#             for deg_th in deg_th_list:
#                 time_subtask_start = time.perf_counter()
#                 os.system('python ./examples/pytorch/delta_gnn/gen_mem_trace.py --dataset=' + dataset +
#                           '--arch=' + arch + '--deg-threshold=' + str(deg_th) + ' | tee -a ' +
#                           log_path)
#                 print('\n>> SubTask {:s} @ {:s} deg_th={:s} exe time: {:s}'.format(
#                     arch, dataset, str(deg_th),
#                     util.time_format(time.perf_counter() - time_subtask_start)))
#         else:
#             time_subtask_start = time.perf_counter()
#             os.system('python ./examples/pytorch/delta_gnn/gen_mem_trace.py --dataset=' + dataset +
#                       '--arch=' + arch + ' | tee -a ' + log_path)
#             print('\n>> SubTask {:s} @ {:s} exe time: {:s}'.format(
#                 arch, dataset, util.time_format(time.perf_counter() - time_subtask_start)))

for dataset in dataset_list:
    for arch in arch_list:
        if arch == 'delta-gnn' or arch == 'delta-gnn-opt':
            if dataset == 'cora' or dataset == 'citeseer':
                deg_th_list = deg_th_list_small_dataset
            elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
                deg_th_list = deg_th_list_large_dataset
            for deg_th in deg_th_list:
                time_subtask_start = time.perf_counter()
                os.system('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset +
                          ' --arch=' + arch + ' --deg-threshold=' + str(deg_th) + ' | tee -a ' +
                          log_path)
                print('\n>> SubTask {:s} @ {:s} deg_th={:s} exe time: {:s}'.format(
                    arch, dataset, str(deg_th),
                    util.time_format(time.perf_counter() - time_subtask_start)))
        else:
            time_subtask_start = time.perf_counter()
            os.system('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset + ' --arch=' +
                      arch + ' | tee -a ' + log_path)
            print('\n>> SubTask {:s} @ {:s} exe time: {:s}'.format(
                arch, dataset, util.time_format(time.perf_counter() - time_subtask_start)))

print('\n>> Task execution time: {:s}'.format(
    util.time_format(time.perf_counter() - Task_time_start)))
