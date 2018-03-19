from model.lib.bbox.nms import torch_nms
from model.lib.bbox.cpu.cython_nms import cython_nms

import random
import numpy as np
import time
import torch


def run_check_nms():

    #test nms:
    H, W = 480, 640
    num_objects = 4
    rois = []
    for n in range(num_objects):
        w = np.random.randint(64, 256)
        h = np.random.randint(64, 256)
        x0 = np.random.randint(0, W-w)
        y0 = np.random.randint(0, H-h)
        x1 = x0 + w
        y1 = y0 + h
        gt = [x0, y0, x1, y1]

        M = np.random.randint(10, 20)
        for m in range(M):
            dw = int(np.random.uniform(0.5, 2)*w)
            dh = int(np.random.uniform(0.5, 2)*h)
            dx = int(np.random.uniform(-1, 1)*w*0.5)
            dy = int(np.random.uniform(-1, 1)*h*0.5)
            xx0 = x0 - dw//2 + dx
            yy0 = y0 - dh//2 + dy
            xx1 = xx0 + w+dw
            yy1 = yy0 + h+dh
            score = np.random.uniform(0.5, 2)

            rois.append([xx0, yy0, xx1, yy1, score])
            pass

    rois = np.array(rois).astype(np.float32)

    start = time.time()
    keep = cython_nms(rois, 0.5)
    print('cython_nms :', keep)
    print("time:", time.time()-start)

    rois = torch.from_numpy(rois)
    start = time.time()
    keep = torch_nms(rois, 0.5)
    print("time:", time.time() - start)
    print('torch_nms.cpu  :', keep.numpy())

    rois2 = rois.cuda()
    start = time.time()
    keep2 = torch_nms(rois2, 0.5)
    print("time:", time.time() - start)
    keep2 = keep2.cpu().numpy()
    print('torch_nms  :', keep2)


if __name__ == '__main__':

    SEED = 35202
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    run_check_nms()
