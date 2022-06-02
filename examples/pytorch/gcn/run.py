import os
import datetime

deg_th_list = [1, 2, 5, 10, 20, 30]
epoch = 200
task_num = len(deg_th_list)

ISOTIMEFORMAT = '%m%d_%H%M'
theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
theTime = str(theTime)

# Predefined
log_path = './results/' + theTime + '_log.txt'

# Remove previous log
os.system('rm ' + log_path)

for task_id in range(task_num):
    # Task batch
    print('\n********** Task id = {:d} **********'.format(task_id))
    # Cora
    deg_th = str(deg_th_list[task_id])
    epoch_str = str(epoch)

    os.system('python ./examples/pytorch/gcn/train_test.py --dataset=cora --n-epochs=' + epoch_str +
              ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
    # Plot accuracy
    os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # Plot retrain cost
    os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

    # # Citeseer
    # os.system('python ./examples/pytorch/gcn/train_test.py --dataset=citeseer --n-epochs=' +
    #           epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
    # # Plot accuracy
    # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # # Plot retrain cost
    # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')

    # # Pubmed
    # os.system('python ./examples/pytorch/gcn/train_test.py --dataset=pubmed --n-epochs=' +
    #           epoch_str + ' --deg-threshold=' + deg_th + ' | tee -a ' + log_path)
    # # Plot accuracy
    # os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')
    # # Plot retrain cost
    # os.system('python ./examples/pytorch/gcn/plt_retrain_cost.py')