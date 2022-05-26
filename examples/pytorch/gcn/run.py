import os

# Task batch
os.system(
    'python ./examples/pytorch/gcn/train_test.py --dataset=cora --n-epochs=200 --deg-threshold=10')

# Plot accuracy
os.system('python ./examples/pytorch/gcn/plt_acc_decline.py')