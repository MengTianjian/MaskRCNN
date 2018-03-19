"""
Training mask r-cnn

"""
from utils.setup import *
from utils.path import train_dir, model_dir, result_dir, log_dir
from model.mask_rcnn import MaskRCNN
from dataset.dataset import CellDataset, train_collate
from model.loss import total_loss

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam, SGD
import torch.nn as nn
import torch
import time
from config import Config

# configurations
lr = Config.LEARNING_RATE
mom = Config.LEARNING_MOMENTUM
num_epoch = 10
batch_size = Config.IMAGES_PER_GPU
print_freq = 10

# data transformations
#transform = transforms.Compose([
#    transforms.Resize(),
#    transforms.ToTensor(),
#    transforms.Normalize()
#])

config = Config()
# load data
train_dataset = CellDataset(train_dir, config)#, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=train_collate)

# model setup
model = MaskRCNN(config).cuda()

#optimizer = Adam(model.parameters(), lr=lr)
optimizer = SGD(model.parameters(), lr=lr, momentum=mom)

for epoch in range(num_epoch):
    running_loss = 0
    end = time.time()
    for i, (imgs, gts) in enumerate(train_loader):
        imgs = imgs.float().cuda()
        data_time = time.time() - end

        # compute loss
        logits = model.forward(imgs)

        loss, saved_for_log = total_loss(logits, gts, config)

        # learn & update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print logs
        running_loss += saved_for_log['total_loss']
        rpn_loss = saved_for_log['rpn_cls_loss']+saved_for_log['rpn_reg_loss']
        mask_loss = saved_for_log['stage2_mask_loss']
        rcnn_loss = saved_for_log['stage2_cls_loss']+saved_for_log['stage2_reg_loss']
        batch_time = time.time() - end
        end = time.time()

        if i % print_freq == print_freq-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Rpn {rpn:.4f}\t'
                  'Rcnn {rcnn:.4f}\t'
                  'Mask {mask:.4f}\t'.format(
                epoch+1, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=running_loss/print_freq,
                rpn=rpn_loss,
                mask=mask_loss,
                rcnn=rcnn_loss))
            running_loss = 0

