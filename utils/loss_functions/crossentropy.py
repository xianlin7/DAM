from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

def cross_entropy_loss(input, target):
    # input: (batch, 2, h, w)
    # target: (batch, h, w)
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return F.cross_entropy(input, target, weight=weight)

