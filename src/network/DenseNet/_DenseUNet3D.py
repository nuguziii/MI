#####################################################
# H-DenseUNet Implementation
# reference: https://arxiv.org/abs/1709.07330
#
# Author:   Chae Eun Lee
# Email:    nuguziii@cglab.snu.ac.kr
#####################################################

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

class Scale3d(nn.Module):
    def __init__(self, num_feature):
        super(Scale3d, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)
    def forward(self, x):
        y = torch.zeros(x.shape, dtype= x.dtype, device= x.device)
        for i in range(self.num_feature):
            y[:, i, :, :, :] = x[:, i, :, :, :].clone() * self.gamma[i] + self.beta[i]
        return y


class conv_block3d(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block3d, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm3d(nb_inp_fea, eps=eps, momentum=1))
        self.add_module('scale1', Scale3d(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv3d1', nn.Conv3d(nb_inp_fea, 4 * growth_rate, (1, 1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm3d(4 * growth_rate, eps=eps, momentum=1))
        self.add_module('scale2', Scale3d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv3d2', nn.Conv3d(4 * growth_rate, growth_rate, (3, 3, 3), padding=(1, 1, 1), bias=False))

    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv3d1(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv3d2(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        return out


class dense_block3d(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block3d, self).__init__()
        for i in range(nb_layers):
            layer = conv_block3d(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer3d%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class _Transition3d(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input))
        self.add_module('scale', Scale3d(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv3d', nn.Conv3d(num_input, num_output, (1, 1, 1), bias=False))
        if (drop > 0):
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

class DenseUNet3D(nn.Module):
    def __init__(self, num_input, growth_rate=32, nb_layers=(3, 4, 12, 8), nb_filter=96, drop_rate=0, nb_classes=2):
        super(DenseUNet3D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(num_input, nb_filter, kernel_size=7, stride=2,
                                             padding=3, bias=False),
                                   nn.BatchNorm3d(nb_filter, eps=1.1e-5),
                                   Scale3d(nb_filter),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        self.dense_block1 = dense_block3d(nb_layers[0], nb_filter, growth_rate, drop_rate)
        nb_filter += nb_layers[0] * growth_rate

        self.transition_block1 = _Transition3d(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.dense_block2 = dense_block3d(nb_layers[1], nb_filter, growth_rate, drop_rate)
        nb_filter += nb_layers[1] * growth_rate

        self.transition_block2 = _Transition3d(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.dense_block3 = dense_block3d(nb_layers[2], nb_filter, growth_rate, drop_rate)
        nb_filter += nb_layers[2] * growth_rate

        self.transition_block3 = _Transition3d(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.dense_block4 = dense_block3d(nb_layers[3], nb_filter, growth_rate, drop_rate)
        nb_filter += nb_layers[3] * growth_rate

        self.block = nn.Sequential(nn.BatchNorm3d(nb_filter, eps=1.1e-5, momentum = 1),
                                    Scale3d(nb_filter),
                                    nn.ReLU(inplace=True))

        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=(1,2,2)),
                                       nn.Conv3d(nb_filter, 504, (3, 3, 3), padding= 1),
                                       nn.BatchNorm3d(504, momentum= 1),
                                       nn.ReLU(inplace=True))

        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=(1,2,2)),
                                       nn.Conv3d(504, 224, (3, 3, 3), padding=1),
                                       nn.BatchNorm3d(224, momentum=1),
                                       nn.ReLU(inplace=True))

        self.upsample3 = nn.Sequential(nn.Upsample(scale_factor=(1,2,2)),
                                       nn.Conv3d(224, 192, (3, 3, 3), padding=1),
                                       nn.BatchNorm3d(192, momentum=1),
                                       nn.ReLU(inplace=True))

        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=(2,2,2)),
                                       nn.Conv3d(192, 96, (3, 3, 3), padding=1),
                                       nn.BatchNorm3d(96, momentum=1),
                                       nn.ReLU(inplace=True))

        self.upsample5 = nn.Sequential(nn.Upsample(scale_factor=(2,2,2)),
                                       nn.Conv3d(96, 64, (3, 3, 3), padding=1),
                                       nn.BatchNorm3d(64, momentum=1),
                                       nn.ReLU(inplace=True))

        self.conv2 = nn.Conv3d(64, nb_classes, (1,1,1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.transition_block1(x)
        x = self.dense_block2(x)
        x = self.transition_block2(x)
        x = self.dense_block3(x)
        x = self.transition_block3(x)
        x = self.dense_block4(x)
        x = self.block(x)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        x_feature = x.clone()
        x = self.conv2(x)
        return x, x_feature

if __name__ == "__main__":
    from torchsummary import summary
    model = DenseUNet3D(num_input=3, growth_rate=32, nb_layers=(3, 4, 12, 8), nb_filter=96, drop_rate=0, nb_classes=2)
    model.cuda()

    summary(model, input_size=(3, 12, 224, 224), batch_size=1)