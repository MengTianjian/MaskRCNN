from model.lib.bbox.cpu.cython_bbox import bbox_overlaps

import numpy as np
import torch


def torch_box_overlap(boxes, gt_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and gt_boxes
    """

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    intersect_ws = (
        torch.min(boxes[:, 2:3], gt_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], gt_boxes[:, 0:1].t()) + 1).clamp(
        min=0)
    intersect_hs = (
        torch.min(boxes[:, 3:4], gt_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], gt_boxes[:, 1:2].t()) + 1).clamp(
        min=0)
    intersect_areas = intersect_ws * intersect_hs
    union_areas = box_areas.view(-1, 1) + gt_areas.view(1, -1) - intersect_areas
    overlaps = intersect_areas / union_areas

    return overlaps


def box_overlap(boxes, gt_boxes):
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    intersect_ws = np.clip(
        (np.min(boxes[:, 2:3], gt_boxes[:, 2:3].t()) - np.max(boxes[:, 0:1], gt_boxes[:, 0:1].t()) + 1), 0, 1e8)
    intersect_hs = np.clip(
        (np.min(boxes[:, 3:4], gt_boxes[:, 3:4].t()) - np.max(boxes[:, 1:2], gt_boxes[:, 1:2].t()) + 1), 0, 1e8)
    intersect_areas = intersect_ws * intersect_hs
    union_areas = box_areas.view(-1, 1) + gt_areas.view(1, -1) - intersect_areas
    overlaps = intersect_areas / union_areas

    return overlaps


if __name__ == '__main__':
    # <todo>
    raise NotImplementedError
