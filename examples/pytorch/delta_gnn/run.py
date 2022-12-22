import os
import datetime
import time
import util
import subprocess

deg_th_list_small_dataset = [1, 2, 5, 10, 20, 30]
# deg_th_list_large_dataset = [50, 100, 300, 500, 1000, 2000]
deg_th_list_large_dataset = [10, 20, 30, 50, 100]
# deg_th_list_large_dataset = [10, 20]
arch_list = ['gcn', 'graphsage', 'gat']
dataset_list = ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-mag']
# dataset_list = ['ogbn-arxiv', 'ogbn-mag']
epoch = 200
# task_num = len(deg_th_list)

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

Task_time_start = time.perf_counter()

# Predefined
log_path = '../../../results/' + theTime + '_log.txt'

# Remove previous log
os.system('rm ' + log_path)

# os.system('cd ./examples/pytorch/delta_gnn/')

for dataset in dataset_list:
    for arch in arch_list:
        # os.system('pwd')
        if dataset == 'cora' or dataset == 'citeseer':
            epoch = 200
            os.system('/usr/bin/python3.8 ./train_evo.py' + ' --dataset=' + dataset + ' --model=' +
                      arch + ' --n-epochs=' + str(epoch))

            for deg_th in deg_th_list_small_dataset:
                os.system('/usr/bin/python3.8 ./train_evo_delta.py' + ' --dataset=' + dataset +
                          ' --model=' + arch + ' --deg-threshold=' + str(deg_th) + ' --n-epochs=' +
                          str(epoch))
        elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-mag':
            epoch = 100
            os.system('/usr/bin/python3.8 ./train_evo.py' + ' --dataset=' + dataset + ' --model=' +
                      arch + ' --n-epochs=' + str(epoch))
            for deg_th in deg_th_list_large_dataset:
                os.system('/usr/bin/python3.8 ./train_evo_delta.py' + ' --dataset=' + dataset +
                          ' --model=' + arch + ' --deg-threshold=' + str(deg_th) + ' --n-epochs=' +
                          str(epoch))

# # Cora
# # os.system('/usr/bin/python3.8 ./train_evo.py --dataset=cora --n-epochs=' + epoch_str +
# #           ' | tee -a ' + log_path)
# subprocess.Popen('/usr/bin/python3.8 ./train_evo.py --dataset=cora --n-epochs=' + epoch_str)

# # Citeseer
# # os.system('/usr/bin/python3.8 ./train_evo.py --dataset=citeseer --n-epochs=' + epoch_str +
# #           ' | tee -a ' + log_path)
# subprocess.Popen('/usr/bin/python3.8 ./train_evo.py --dataset=citeseer --n-epochs=' + epoch_str +
#                  ' | tee -a ' + log_path)

# # Pubmed
# # os.system('/usr/bin/python3.8 ./train_evo.py --dataset=pubmed --n-epochs=' + epoch_str +
# #           ' | tee -a ' + log_path)
# subprocess.Popen('/usr/bin/python3.8 ./train_evo.py --dataset=pubmed --n-epochs=' + epoch_str +
#                  ' | tee -a ' + log_path)

# # # Amazon Computer
# # os.system('python ./examples/pytorch/gcn/train_evo.py --dataset=amazon_comp --n-epochs=' +
# #           epoch_str + ' | tee -a ' + log_path)

# # # Plot accuracy
# # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
# # # Plot retrain cost
# # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

# for task_id in range(task_num):
#     # Task batch
#     print('\n********** Task id = {:d} **********'.format(task_id))

#     deg_th = str(deg_th_list[task_id])
#     deg_th_large = str(deg_th_list_large_dataset[task_id])

#     # Cora
#     # os.system('/usr/bin/python3.8 ./train_evo_delta.py --dataset=cora --n-epochs=' + epoch_str +
#     #           ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
#     subprocess.Popen('/usr/bin/python3.8 ./train_evo_delta.py --dataset=cora --n-epochs=' +
#                      epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)

#     # # Plot accuracy
#     # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
#     # # Plot retrain cost
#     # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

#     # Citeseer
#     # os.system('/usr/bin/python3.8 ./train_evo_delta.py --dataset=citeseer --n-epochs=' + epoch_str +
#     #           ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
#     subprocess.Popen('/usr/bin/python3.8 ./train_evo_delta.py --dataset=citeseer --n-epochs=' +
#                      epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)

#     # # Plot accuracy
#     # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
#     # # Plot retrain cost
#     # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

#     # Pubmed
#     # os.system('/usr/bin/python3.8 ./train_evo_delta.py --dataset=pubmed --n-epochs=' + epoch_str +
#     #           ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
#     subprocess.Popen('/usr/bin/python3.8 ./train_evo_delta.py --dataset=pubmed --n-epochs=' +
#                      epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)

#     # # Plot accuracy
#     # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
#     # # Plot retrain cost
#     # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

#     # # Amazon Computer
#     # os.system('python ./examples/pytorch/gcn/train_evo_delta.py --dataset=amazon_comp --n-epochs=' +
#     #           epoch_str + ' --deg-threshold=' + deg_th_large + ' | tee -a ' + log_path)

#     # # Plot accuracy
#     # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
#     # # Plot retrain cost
#     # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

print('\n>> All Tasks finish, total execution time: {:.4}s'.format(
    util.time_format(time.perf_counter() - Task_time_start)))
