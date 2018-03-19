"""
Check CUDA & Pytorch version

"""
import os
import torch

print('\tset cuda environment')
print('\t\ttorch.__version__              =', torch.__version__)
print('\t\ttorch.version.cuda             =', torch.version.cuda)
print('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())

# how many GPUs do you have
try:
    print('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
    NUM_CUDA_DEVICES = 1

print('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
print('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())
