#####################################################
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################

import torch.nn as nn
from torch.nn import functional as F

class loss_ce(nn.Module):
    def __init__(self):
        super(loss_ce, self).__init__()

    def forward(self, input, target, weights=None, ignore=255):
        return F.cross_entropy(
            input.float(),
            target.long(),
            ignore_index=ignore,
            weight=weights,
        )

class loss_l1(nn.Module):
    def __init__(self):
        super(loss_l1, self).__init__()

    def forward(self, input, target):
        return F.l1_loss(input, target)