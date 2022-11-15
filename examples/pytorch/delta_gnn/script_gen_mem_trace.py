import os
import datetime
import time
import util
import subprocess

# deg_th_list_small_dataset = [1, 2, 5, 10, 20, 30]
deg_th_list_small_dataset = [1, 2, 5, 10, 15, 20, 30, 50, 100]
deg_th_list_large_dataset = [1, 5, 10, 20, 50, 100]
# epoch = 200
# task_num = len(deg_th_list)
# arch_list = ['hygcn', 'awb-gcn', 'i-gcn', 'regnn', 'delta-gnn', 'delta-gnn-opt']
arch_list = ['delta-gnn']
# dataset_list = ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-mag']
dataset_list = ['cora']

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
        if dataset == 'cora' or dataset == 'citeseer':
            sample_nodes_num = 50
        elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
            sample_nodes_num = 2

        if arch == 'delta-gnn' or arch == 'delta-gnn-opt':
            if dataset == 'cora' or dataset == 'citeseer':
                deg_th_list = deg_th_list_small_dataset
            elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
                deg_th_list = deg_th_list_large_dataset
            for deg_th in deg_th_list:
                time_subtask_start = time.perf_counter()
                # os.system('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset +
                #           ' --arch=' + arch + ' --deg-threshold=' + str(deg_th) +
                #           ' --sample-node-num=' + str(sample_nodes_num) + ' | tee -a ' + log_path)
                subprocess.Popen('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset +
                                 ' --arch=' + arch + ' --deg-threshold=' + str(deg_th) +
                                 ' --sample-node-num=' + str(sample_nodes_num),
                                 shell=True)
                print('\n>> SubTask {:s} @ {:s} deg_th={:s} exe time: {:s}'.format(
                    arch, dataset, str(deg_th),
                    util.time_format(time.perf_counter() - time_subtask_start)))
        else:
            time_subtask_start = time.perf_counter()
            # os.system('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset + ' --arch=' +
            #           arch + ' --sample-node-num=' + str(sample_nodes_num) + ' | tee -a ' +
            #           log_path)
            subprocess.Popen('/usr/bin/python3.8 ./gen_mem_trace.py --dataset=' + dataset +
                             ' --arch=' + arch + ' --sample-node-num=' + str(sample_nodes_num),
                             shell=True)
            print('\n>> SubTask {:s} @ {:s} exe time: {:s}'.format(
                arch, dataset, util.time_format(time.perf_counter() - time_subtask_start)))

print('\n>> All Tasks have completed, total execution time: {:s}'.format(
    util.time_format(time.perf_counter() - Task_time_start)))
