#####################################################
#
# reference: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
#
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################
import torch.nn as nn

class loss_softdice(nn.Module):
    def __init__(self):
        super(loss_softdice, self).__init__()

    def forward(self, input, target, weights=None):
        '''
            :param target: NxDxHxW LongTensor
            :param output: NxCxDxHxW Variable
            :param weights: C FloatTensor
            :param ignore_index: int index to ignore from loss
            :return:
            '''
        eps = 0.0001
        encoded_target = input.detach() * 0

        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = input * encoded_target
        numerator = 2 * intersection.sum(2).sum(2).sum(2)
        denominator = input + encoded_target
        denominator = denominator.sum(2).sum(2).sum(2) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / loss_per_channel.size(0)

class loss_dice(nn.Module):
    def __init__(self):
        super(loss_dice, self).__init__()

    def forward(self, input, target):
        eps = 0.0001
        _, output = input.max(1)

        intersection = output * target
        numerator = 2 * intersection.sum(1).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(1).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / loss_per_channel.size(0)