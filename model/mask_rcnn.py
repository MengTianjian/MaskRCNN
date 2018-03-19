"""
Model definitions

"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

from model.resnet import resnet50
from model.rpn import RPN
#from model.lib.roi_align.roi_align.roi_align import RoIAlign
from model.lib.roi_align.roi_align.crop_and_resize import CropAndResize
from model.lib.bbox.generate_anchors import generate_pyramid_anchors
from model.lib.bbox.nms import torch_nms as nms


def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.))


def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.
    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    #feature_maps= [P2, P3, P4, P5]
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)

    roi_number = rois.size()[1]

    pooled = rois.data.new(
            config.IMAGES_PER_GPU*rois.size(
            1), 256, pool_size, pool_size).zero_()

    rois = rois.view(
            config.IMAGES_PER_GPU*rois.size(1),
            4)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    x_1 = rois[:, 0]
    y_1 = rois[:, 1]
    x_2 = rois[:, 2]
    y_2 = rois[:, 3]

    roi_level = log2_graph(
        torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 224.0))

    roi_level = torch.clamp(torch.clamp(
        torch.add(torch.round(roi_level), 4), min=2), max=5)

    # P2 is 256x256, P3 is 128x128, P4 is 64x64, P5 is 32x32
    # P2 is 4, P3 is 8, P4 is 16, P5 is 32
    for i, level in enumerate(range(2, 6)):

        scaling_ratio = 2**level

        height = float(config.IMAGE_MAX_DIM)/ scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio

        ixx = torch.eq(roi_level, level)

        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)

        crops = crop_resize(feature_maps[i], torch.div(
                level_boxes, float(config.IMAGE_MAX_DIM)
                )[:, [1, 0, 3, 2]], box_indices)

        indices_pooled = ixx.nonzero()[:, 0]
        pooled[indices_pooled.data, :, :, :] = crops.data

    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number,
               256, pool_size, pool_size)
    pooled = Variable(pooled).cuda()
    return pooled


# ---------------------------------------------------------------
# Heads

class MaskHead(nn.Module):

    def __init__(self, config):
        super(MaskHead, self).__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        #self.crop_size = config.mask_crop_size

        #self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.mask = nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x, rpn_rois):
        #x = self.roi_align(x, rpn_rois)
        x = ROIAlign(x, rpn_rois, self.config, self.config.MASK_POOL_SIZE)

        roi_number = x.size()[1]

        # merge batch and roi number together
        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.MASK_POOL_SIZE,
                   self.config.MASK_POOL_SIZE)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.deconv(x)
        rcnn_mask_logits = self.mask(x)

        rcnn_mask_logits = rcnn_mask_logits.view(self.config.IMAGES_PER_GPU,
                                                 roi_number,
                                                 self.config.NUM_CLASSES,
                                                 self.config.MASK_POOL_SIZE * 2,
                                                 self.config.MASK_POOL_SIZE * 2)

        return rcnn_mask_logits


class RCNNHead(nn.Module):
    def __init__(self, config):
        super(RCNNHead, self).__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        #self.crop_size = config.rcnn_crop_size

        #self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.class_logits = nn.Linear(1024, self.num_classes)
        self.bbox = nn.Linear(1024, self.num_classes * 4)

        self.conv1 = nn.Conv2d(256, 1024, kernel_size=self.config.POOL_SIZE, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001)

    def forward(self, x, rpn_rois):
        x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)
        roi_number = x.size()[1]

        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.POOL_SIZE,
                   self.config.POOL_SIZE)
        #print(x.shape)
        #x = self.roi_align(x, rpn_rois, self.config, self.config.POOL_SIZE)
        #x = crops.view(crops.size(0), -1)
        x = self.bn1(self.conv1(x))
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        #x = F.dropout(x, 0.5, training=self.training)
        rcnn_class_logits = self.class_logits(x)
        rcnn_probs = F.softmax(rcnn_class_logits, dim=-1)

        rcnn_bbox = self.bbox(x)

        rcnn_class_logits = rcnn_class_logits.view(self.config.IMAGES_PER_GPU,
                                                   roi_number,
                                                   rcnn_class_logits.size()[-1])

        rcnn_probs = rcnn_probs.view(self.config.IMAGES_PER_GPU,
                                     roi_number,
                                     rcnn_probs.size()[-1])

        rcnn_bbox = rcnn_bbox.view(self.config.IMAGES_PER_GPU,
                                   roi_number,
                                   self.config.NUM_CLASSES,
                                   4)

        return rcnn_class_logits, rcnn_probs, rcnn_bbox


#
# ---------------------------------------------------------------
# Mask R-CNN

class MaskRCNN(nn.Module):
    """
    Mask R-CNN model
    """

    def __init__(self, config):
        super(MaskRCNN, self).__init__()
        self.config = config
        self.__mode = 'train'
        feature_channels = 128
        # define modules (set of layers)
#        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.feature_net = resnet50().cuda()
        #self.rpn_head = RpnMultiHead(cfg,feature_channels)
        self.rpn = RPN(256, len(self.config.RPN_ANCHOR_RATIOS),
                             self.config.RPN_ANCHOR_STRIDE)
        #self.rcnn_crop = CropRoi(cfg, cfg.rcnn_crop_size)
        self.rcnn_head = RCNNHead(config)
        #self.mask_crop = CropRoi(cfg, cfg.mask_crop_size)
        self.mask_head = MaskHead(config)

        self.anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                self.config.RPN_ANCHOR_RATIOS,
                                                self.config.BACKBONE_SHAPES,
                                                self.config.BACKBONE_STRIDES,
                                                self.config.RPN_ANCHOR_STRIDE)
        self.anchors = self.anchors.astype(np.float32)
        self.proposal_count = self.config.POST_NMS_ROIS_TRAINING
        # FPN
        self.fpn_c5p5 = nn.Conv2d(
            512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c4p4 = nn.Conv2d(
            256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c3p3 = nn.Conv2d(
            128 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c2p2 = nn.Conv2d(
            64 * 4, 256, kernel_size=1, stride=1,  padding=0)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.scale_ratios = [4, 8, 16, 32]
        self.fpn_p6 = nn.MaxPool2d(
            kernel_size=1, stride=2, padding=0, ceil_mode=False)

    def forward(self, x):
        # Extract features
        C1, C2, C3, C4, C5 = self.feature_net(x)
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + F.upsample(P5,
                                            scale_factor=2, mode='bilinear')
        P3 = self.fpn_c3p3(C3) + F.upsample(P4,
                                            scale_factor=2, mode='bilinear')
        P2 = self.fpn_c2p2(C2) + F.upsample(P3,
                                            scale_factor=2, mode='bilinear')

        # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 is 256, P3 is 128, P4 is 64, P5 is 32
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]

        self.mrcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits_outputs = []
        rpn_class_outputs = []
        rpn_bbox_outputs = []
        # RPN proposals
        for feature in rpn_feature_maps:
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(feature)
            rpn_class_logits_outputs.append(rpn_class_logits)
            rpn_class_outputs.append(rpn_probs)
            rpn_bbox_outputs.append(rpn_bbox)

        rpn_class_logits = torch.cat(rpn_class_logits_outputs, dim=1)
        rpn_class = torch.cat(rpn_class_outputs, dim=1)
        rpn_bbox = torch.cat(rpn_bbox_outputs, dim=1)

        rpn_proposals = self.proposal_layer(rpn_class, rpn_bbox)

        # RCNN proposals
        rcnn_class_logits, rcnn_class, rcnn_bbox = self.rcnn_head(self.mrcnn_feature_maps, rpn_proposals)
        rcnn_mask_logits = self.mask_head(self.mrcnn_feature_maps, rpn_proposals)
        # <todo> mask nms

        return [rpn_class_logits, rpn_class, rpn_bbox, rpn_proposals,
                rcnn_class_logits, rcnn_class, rcnn_bbox,
                rcnn_mask_logits]

    def proposal_layer(self, rpn_class, rpn_bbox):
        # handling proposals
        scores = rpn_class[:, :, 1]
        #print(scores.shape)
        # Box deltas [batch, num_rois, 4]
        deltas_mul = Variable(torch.from_numpy(np.reshape(
            self.config.RPN_BBOX_STD_DEV, [1, 1, 4]).astype(np.float32))).cuda()
        deltas = rpn_bbox * deltas_mul

        pre_nms_limit = min(6000, self.anchors.shape[0])

        scores, ix = torch.topk(scores, pre_nms_limit, dim=-1,
                                largest=True, sorted=True)

        ix = torch.unsqueeze(ix, 2)
        ix = torch.cat([ix, ix, ix, ix], dim=2)
        deltas = torch.gather(deltas, 1, ix)

        _anchors = []
        for i in range(self.config.IMAGES_PER_GPU):
            anchors = Variable(torch.from_numpy(
                self.anchors.astype(np.float32))).cuda()
            _anchors.append(anchors)
        anchors = torch.stack(_anchors, 0)

        pre_nms_anchors = torch.gather(anchors, 1, ix)
        refined_anchors = apply_box_deltas_graph(pre_nms_anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        window = Variable(torch.from_numpy(window)).cuda()

        refined_anchors_clipped = clip_boxes_graph(refined_anchors, window)

        refined_proposals = []
        scores = scores[:,:,None]
        #print(scores.data.shape)
        #print(refined_anchors_clipped.data.shape)
        for i in range(self.config.IMAGES_PER_GPU):
            indices = nms(
                torch.cat([refined_anchors_clipped.data[i], scores.data[i]], 1), 0.7)
            indices = indices[:self.proposal_count]
            indices = torch.stack([indices, indices, indices, indices], dim=1)
            indices = Variable(indices).cuda()
            proposals = torch.gather(refined_anchors_clipped[i], 0, indices)
            padding = self.proposal_count - proposals.size()[0]
            proposals = torch.cat(
                [proposals, Variable(torch.zeros([padding, 4])).cuda()], 0)
            refined_proposals.append(proposals)

        rpn_rois = torch.stack(refined_proposals, 0)

        return rpn_rois


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, :, 0] * height
    center_x += deltas[:, :, 1] * width
    height *= torch.exp(deltas[:, :, 2])
    width *= torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = [y1, x1, y2, x2]
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = window
    y1, x1, y2, x2 = boxes
    # Clip

    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)

    clipped = torch.stack([x1, y1, x2, y2], dim=2)
    return clipped
