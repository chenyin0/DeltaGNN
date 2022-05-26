import os
import time

# Task batch
time_start = time.perf_counter()
os.system('python ./examples/pytorch/gcn/train_test.py --dataset=cora --n-epochs=200 --deg-threshold=20')
print('\n>> Task Cora execution time: {:.4}s'.format(time.perf_counter() - time_start))
# Plot accuracy
os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')

# time_start = time.perf_counter()
# os.system('python ./examples/pytorch/gcn/train_test.py --dataset=citeseer --n-epochs=200 --deg-threshold=5')
# print('\n>> Task Citeseer execution time: {:.4}s'.format(time.perf_counter() - time_start))
# # Plot accuracy
# os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')

# time_start = time.perf_counter()
# os.system('python ./examples/pytorch/gcn/train_test.py --dataset=pubmed --n-epochs=200 --deg-threshold=5')
# print('\n>> Task Pubmed execution time: {:.4}s'.format(time.perf_counter() - time_start))
# # Plot accuracy
# os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')