import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

class CrossEntropy2d_LL(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d_LL, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict_all, target_all, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target_all.requires_grad
        assert predict_all.dim() == 4
        assert target_all.dim() == 3
        n, c, h, w = predict_all.size()
        loss = torch.zeros(n,1).cuda()
        for i in range(n):
            target = torch.zeros(1, h, w).long().cuda()
            predict = torch.zeros(1, c, h, w).float().cuda()
            target[0] = target_all[i].cuda()
            predict[0] = predict_all[i].cuda()
            
            target_mask = (target >= 0) * (target != self.ignore_label)
            target = target[target_mask].cuda()
            if not target.data.dim():
                return Variable(torch.zeros(1).cuda())
            predict = predict.transpose(1, 2).transpose(2, 3).contiguous().cuda()
            predict = predict[target_mask.view(1, h, w, 1).repeat(1, 1, 1, c)].view(-1, c).cuda()
            loss[i] = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

