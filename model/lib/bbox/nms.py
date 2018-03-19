import torch
from model.lib.bbox.torch_nms._ext import nms as _backend


def torch_nms(dets, thresh):
    """
    dets has to be a tensor
    """
    if not dets.is_cuda:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        _backend.cpu_nms(keep, num_out, dets, order, areas, thresh)

        return keep[:num_out[0]]

    else:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        order = scores.sort(0, descending=True)[1]

        dets = dets[order].contiguous()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)

        _backend.gpu_nms(keep, num_out, dets, thresh)

        return order[keep[:num_out[0]].cuda()].contiguous()
