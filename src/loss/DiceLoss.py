#####################################################
#
# reference: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
#
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################

def loss_softdice(target, output, weights=None):
    '''
    :param target: NxDxHxW LongTensor
    :param output: NxCxDxHxW Variable
    :param weights: C FloatTensor
    :param ignore_index: int index to ignore from loss
    :return:
    '''
    eps = 0.0001
    encoded_target = output.detach() * 0

    encoded_target.scatter_(1, target.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = output * encoded_target
    numerator = 2 * intersection.sum(2).sum(2).sum(2)
    denominator = output + encoded_target
    denominator = denominator.sum(2).sum(2).sum(2) + eps
    loss_per_channel = weights * (1 - (numerator / denominator))

    return loss_per_channel.sum() / loss_per_channel.size(0)


def loss_dice(target, output):
    eps = 0.0001
    _, output = output.max(1)

    intersection = output * target
    numerator = 2 * intersection.sum(1).sum(1).sum(1)
    denominator = output + target
    denominator = denominator.sum(1).sum(1).sum(1) + eps
    loss_per_channel = 1 - (numerator / denominator)

    return loss_per_channel.sum() / loss_per_channel.size(0)