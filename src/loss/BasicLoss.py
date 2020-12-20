#####################################################
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################

from torch.nn import functional as F

def loss_ce(input, target, weights, ignore=255):
    return F.cross_entropy(
                input.float(),
                target.long(),
                ignore_index=ignore,
                weight=weights,
            )

def loss_l1(input, target):
    return F.l1_loss(input, target)