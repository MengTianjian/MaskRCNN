"""
Pre-process dataset

"""
import os
import numpy as np
from utils.path import data_dir, train_dir
from dataset.dataset import PrepareDataset
import matplotlib.pyplot as plt
import skimage.io as skio


# mkdir if not exists
if not os.path.exists(train_dir + '/images'):
    os.makedirs(train_dir + '/images')
if not os.path.exists(train_dir + '/multi_masks'):
    os.makedirs(train_dir + '/multi_masks')

dataset = PrepareDataset(data_dir+'/stage1_train')

# preprocess training images & generate masks
for img_id, image, multi_mask, gt_mask in dataset:
    skio.imsave(train_dir + '/images/%s.png' % img_id, image)
    plt.imsave(train_dir + '/images/%s_mask.png' % img_id, multi_mask)
    np.save(train_dir + '/multi_masks/%s.npy' % img_id, gt_mask)
