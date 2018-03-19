"""
Setup project settings

"""
import numpy as np
import random
import torch


# ---------------------------------------------------------------------------------
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------------
# setup random seed

SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print('\tset random seed')
print('\t\tSEED=%d' % SEED)


# CUDA & torch setup
torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled = True
