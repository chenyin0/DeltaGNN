import os
import time

deg_th_list = [1, 2, 5, 10, 20, 30]
epoch = 200
task_num = len(deg_th_list)

# Remove previous log
os.system('rm ./results/log.txt')

for task_id in range(task_num):
    # Task batch
    # Cora
    deg_th = str(deg_th_list[task_id])
    epoch_str = str(epoch)

    time_start = time.perf_counter()
    os.system('python ./examples/pytorch/gcn/train_test.py --dataset=cora --n-epochs=' + epoch_str +
              ' --deg-threshold=' + deg_th + ' | tee -a ./results/log.txt')
    print('\n>> Task Cora execution time: {:.4}s'.format(time.perf_counter() - time_start))
    # Plot accuracy
    os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # Plot retrain cost
    os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

    # Citeseer
    time_start = time.perf_counter()
    os.system('python ./examples/pytorch/gcn/train_test.py --dataset=citeseer --n-epochs=' +
              epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ./results/log.txt')
    print('\n>> Task Citeseer execution time: {:.4}s'.format(time.perf_counter() - time_start))
    # Plot accuracy
    os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # Plot retrain cost
    os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

    # Pubmed
    time_start = time.perf_counter()
    os.system('python ./examples/pytorch/gcn/train_test.py --dataset=pubmed --n-epochs=' +
              epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ./results/log.txt')
    print('\n>> Task Pubmed execution time: {:.4}s'.format(time.perf_counter() - time_start))
    # Plot accuracy
    os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # Plot retrain cost
    os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')