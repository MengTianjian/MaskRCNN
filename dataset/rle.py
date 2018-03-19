"""
Utility functions for RLE coding.

"""
import numpy as np
import pandas as pd
from utils.path import *


def rle_encode(mask):
    """ Ref. https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    """
    pixels = mask.flatten('F')
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_str, mask_shape, mask_dtype):
    """

    :param rle_str:
    :param mask_shape:
    :param mask_dtype:
    :return:
    """
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for low, high in zip(starts, ends):
        mask[low:high] = 1
    return mask.reshape(mask_shape[::-1]).T


def rle_decode_location(rle_str):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    return zip(starts, ends)


def read_train_rles(csv_path):
    train_rles_dict = {}
    train_labels = pd.read_csv(csv_path)

    for row in train_labels.itertuples():
        img_id = row[1]
        rle = row[2]

        assert len(img_id) > 0 and len(rle) > 0

        if img_id not in train_rles_dict:
            rles = []
            train_rles_dict[img_id] = rles
        else:
            rles = train_rles_dict.get(img_id)

        rles.append(rle)

    print("Read train RLEs for ", len(train_rles_dict), "images.")
    return train_rles_dict
