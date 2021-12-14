import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision import models
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, in_planes=16, no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelAttention(gate_channels, in_planes)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialAttention()
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         return x_out

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)) # inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s         # kernel_size, ?, ?, use_se, use_hs, stride
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],     # stride = 2, layer -> 1
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],     # stride = 2, layer -> 3
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],     # stride = 2, layer -> 6
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],     # stride = 2, layer -> 12
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],     # stride = 2, layer -> 0
        [3,  4.5,  24, 0, 0, 2],     # stride = 2, layer -> 1
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],     # stride = 2, layer -> 3
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],     # stride = 2, layer -> 8
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def mobilenet_v3_module(pretrain=True,size='large',freeze=True):
    """Constructs a mobilenet_v3 model.
    """
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    if size == 'large':
        model = mobilenetv3_large()
        if pretrain:
            model.load_state_dict(torch.load('utils/model/mobilenetv3-large-1cd25616.pth'))
    elif size == 'small':
        model = mobilenetv3_small()
        if pretrain:
            model.load_state_dict(torch.load('utils/model/mobilenetv3-small-55df8e1f.pth'))
    
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model

def deep_orientation_gen(ins):
    # Up and Down
    ins_U = ins
    ins_D = ins
    # Left and Right
    ins_L = ins
    ins_R = ins
    # Left-Up and Right-Down
    ins_LU = ins
    ins_RD = ins
    # Right-Up and Left-Down
    ins_RU = ins
    ins_LD = ins

    batch_size_tensor, c, w, h = ins.size()

    # Up
    U_ones = torch.ones([1, w], dtype=torch.int)
    U_ones = U_ones.unsqueeze(-1)
    U_range = torch.arange(w, dtype=torch.int).unsqueeze(0)
    U_range = U_range.unsqueeze(1)
    U_channel = torch.matmul(U_ones, U_range)
    U_channel = U_channel.unsqueeze(-1)
    U_channel = U_channel.permute(0, 3, 1, 2)
    U_channel = U_channel.float() / (w - 1)
    U_channel = U_channel * 2 - 1
    U_channel = U_channel.repeat(batch_size_tensor, 1, 1, 1)
    U_channel = U_channel.cuda()
    ins_U_new = torch.cat((ins_U, U_channel), 1)

    # Down
    D_ones = torch.ones([1, w], dtype=torch.int)
    D_ones = D_ones.unsqueeze(-1)
    D_range = torch.arange(w - 1, -1, -1, dtype=torch.int).unsqueeze(0)
    D_range = D_range.unsqueeze(1)
    D_channel = torch.matmul(D_ones, D_range)
    D_channel = D_channel.unsqueeze(-1)
    D_channel = D_channel.permute(0, 3, 1, 2)
    D_channel = D_channel.float() / (w - 1)
    D_channel = D_channel * 2 - 1
    D_channel = D_channel.repeat(batch_size_tensor, 1, 1, 1)
    D_channel = D_channel.cuda()
    ins_D_new = torch.cat((ins_D, D_channel), 1)

    # Left
    L_ones = torch.ones([1, h], dtype=torch.int)
    L_ones = L_ones.unsqueeze(1)
    L_range = torch.arange(h, dtype=torch.int).unsqueeze(0)
    L_range = L_range.unsqueeze(-1)
    L_channel = torch.matmul(L_range, L_ones)
    L_channel = L_channel.unsqueeze(-1)
    L_channel = L_channel.permute(0, 3, 1, 2)
    L_channel = L_channel.float() / (h - 1)
    L_channel = L_channel * 2 - 1
    L_channel = L_channel.repeat(batch_size_tensor, 1, 1, 1)
    L_channel = L_channel.cuda()
    ins_L_new = torch.cat((ins_L, L_channel), 1)

    # Right
    R_ones = torch.ones([1, h], dtype=torch.int)
    R_ones = R_ones.unsqueeze(1)
    R_range = torch.arange(h - 1, -1, -1, dtype=torch.int).unsqueeze(0)
    R_range = R_range.unsqueeze(-1)
    R_channel = torch.matmul(R_range, R_ones)
    R_channel = R_channel.unsqueeze(-1)
    R_channel = R_channel.permute(0, 3, 1, 2)
    R_channel = R_channel.float() / (h - 1)
    R_channel = R_channel * 2 - 1
    R_channel = R_channel.repeat(batch_size_tensor, 1, 1, 1)
    R_channel = R_channel.cuda()
    ins_R_new = torch.cat((ins_R, R_channel), 1)

    # Left and Up
    LU_ones_1 = torch.ones([w, h], dtype=torch.int)
    LU_ones_1 = torch.triu(LU_ones_1)
    LU_ones_2 = torch.ones([w, h], dtype=torch.int)
    LU_change = torch.arange(h - 1, -1, -1, dtype=torch.int)
    LU_ones_2[w - 1, :] = LU_change
    LU_channel = torch.matmul(LU_ones_1, LU_ones_2)
    LU_channel = LU_channel.unsqueeze(0).unsqueeze(-1)
    LU_channel = LU_channel.permute(0, 3, 1, 2)
    LU_channel = LU_channel.float() / (h - 1)
    LU_channel = LU_channel * 2 - 1
    LU_channel = LU_channel.repeat(batch_size_tensor, 1, 1, 1)
    LU_channel = LU_channel.cuda()
    ins_LU_new = torch.cat((ins_LU, LU_channel), 1)

    # Right and Down
    RD_ones_1 = torch.ones([w, h], dtype=torch.int)
    RD_ones_1 = torch.triu(RD_ones_1)
    RD_ones_1 = torch.t(RD_ones_1)
    RD_ones_2 = torch.ones([w, h], dtype=torch.int)
    RD_change = torch.arange(h, dtype=torch.int)
    RD_ones_2[0, :] = RD_change
    RD_channel = torch.matmul(RD_ones_1, RD_ones_2)
    RD_channel = RD_channel.unsqueeze(0).unsqueeze(-1)
    RD_channel = RD_channel.permute(0, 3, 1, 2)
    RD_channel = RD_channel.float() / (h - 1)
    RD_channel = RD_channel * 2 - 1
    RD_channel = RD_channel.repeat(batch_size_tensor, 1, 1, 1)
    RD_channel = RD_channel.cuda()
    ins_RD_new = torch.cat((ins_RD, RD_channel), 1)

    # Right and Up
    RU_ones_1 = torch.ones([w, h], dtype=torch.int)
    RU_ones_1 = torch.triu(RU_ones_1)
    RU_ones_2 = torch.ones([w, h], dtype=torch.int)
    RU_change = torch.arange(h, dtype=torch.int)
    RU_ones_2[w - 1, :] = RU_change
    RU_channel = torch.matmul(RU_ones_1, RU_ones_2)
    RU_channel = RU_channel.unsqueeze(0).unsqueeze(-1)
    RU_channel = RU_channel.permute(0, 3, 1, 2)
    RU_channel = RU_channel.float() / (h - 1)
    RU_channel = RU_channel * 2 - 1
    RU_channel = RU_channel.repeat(batch_size_tensor, 1, 1, 1)
    RU_channel = RU_channel.cuda()
    ins_RU_new = torch.cat((ins_RU, RU_channel), 1)

    # Left and Down
    LD_ones_1 = torch.ones([w, h], dtype=torch.int)
    LD_ones_1 = torch.triu(LD_ones_1)
    LD_ones_1 = torch.t(LD_ones_1)
    LD_ones_2 = torch.ones([w, h], dtype=torch.int)
    LD_change = torch.arange(h - 1, -1, -1, dtype=torch.int)
    LD_ones_2[0, :] = LD_change
    LD_channel = torch.matmul(LD_ones_1, LD_ones_2)
    LD_channel = LD_channel.unsqueeze(0).unsqueeze(-1)
    LD_channel = LD_channel.permute(0, 3, 1, 2)
    LD_channel = LD_channel.float() / (h - 1)
    LD_channel = LD_channel * 2 - 1
    LD_channel = LD_channel.repeat(batch_size_tensor, 1, 1, 1)
    LD_channel = LD_channel.cuda()
    ins_LD_new = torch.cat((ins_LD, LD_channel), 1)

    return ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new


class Deep_Orientation(nn.Module):
    def __init__(self, input_channel, output_channel, mid_channel):
        super(Deep_Orientation, self).__init__()

        self.transition_1 = nn.Conv2d(input_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.transition_1_bn = nn.BatchNorm2d(input_channel)

        self.transition_2_U = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_U_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_D = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_D_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_L = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.transition_2_L_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_R = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                        bias=False, dilation=1)
        self.transition_2_R_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_LU = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_LU_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_RD = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_RD_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_2_RU = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_RU_bn = nn.BatchNorm2d(1 + mid_channel)
        
        self.transition_2_LD = nn.Conv2d(1 + mid_channel, mid_channel // 8, kernel_size=3, stride=1, padding=1,
                                         bias=False, dilation=1)
        self.transition_2_LD_bn = nn.BatchNorm2d(1 + mid_channel)

        self.transition_3 = nn.Conv2d(mid_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.transition_3_bn = nn.BatchNorm2d(mid_channel)

        self.scale = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            # nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

    def forward(self, x, stage=None):
        x = F.relu(self.transition_1(self.transition_1_bn(x)),inplace=True)

        ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new = deep_orientation_gen(x)

        ins_U_new = F.relu(self.transition_2_U(self.transition_2_U_bn(ins_U_new)),inplace=True)
        ins_D_new = F.relu(self.transition_2_D(self.transition_2_D_bn(ins_D_new)),inplace=True)
        ins_L_new = F.relu(self.transition_2_L(self.transition_2_L_bn(ins_L_new)),inplace=True)
        ins_R_new = F.relu(self.transition_2_R(self.transition_2_R_bn(ins_R_new)),inplace=True)
        ins_LU_new = F.relu(self.transition_2_LU(self.transition_2_LU_bn(ins_LU_new)),inplace=True)
        ins_RD_new = F.relu(self.transition_2_RD(self.transition_2_RD_bn(ins_RD_new)),inplace=True)
        ins_RU_new = F.relu(self.transition_2_RU(self.transition_2_RU_bn(ins_RU_new)),inplace=True)
        ins_LD_new = F.relu(self.transition_2_LD(self.transition_2_LD_bn(ins_LD_new)),inplace=True)

        batch = ins_U_new.shape[0]
        scale_U = ins_U_new.reshape((batch, -1)).max(1)[0]
        scale_D = ins_D_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_L = ins_L_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_R = ins_R_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_LU = ins_LU_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_RD = ins_RD_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_RU = ins_RU_new.reshape((batch, -1)).max(1)[0] # ins_U_new
        scale_LD = ins_LD_new.reshape((batch, -1)).max(1)[0] # ins_U_new

        scale_U = scale_U.unsqueeze(1)
        scale_D = scale_D.unsqueeze(1)
        scale_L = scale_L.unsqueeze(1)
        scale_R = scale_R.unsqueeze(1)
        scale_LU = scale_LU.unsqueeze(1)
        scale_RD = scale_RD.unsqueeze(1)
        scale_RU = scale_RU.unsqueeze(1)
        scale_LD = scale_LD.unsqueeze(1)

        scale = torch.cat((scale_U, scale_D, scale_L, scale_R, scale_LU, scale_RD, scale_RU, scale_LD), 1)
        scale = self.scale(scale)

        ins_U_new = scale[:, 0:1].unsqueeze(2).unsqueeze(3) * ins_U_new
        ins_D_new = scale[:, 1:2].unsqueeze(2).unsqueeze(3) * ins_D_new

        ins_L_new = scale[:, 2:3].unsqueeze(2).unsqueeze(3) * ins_L_new
        ins_R_new = scale[:, 3:4].unsqueeze(2).unsqueeze(3) * ins_R_new

        ins_LU_new = scale[:, 4:5].unsqueeze(2).unsqueeze(3) * ins_LU_new
        ins_RD_new = scale[:, 5:6].unsqueeze(2).unsqueeze(3) * ins_RD_new

        ins_RU_new = scale[:, 6:7].unsqueeze(2).unsqueeze(3) * ins_RU_new
        ins_LD_new = scale[:, 7:8].unsqueeze(2).unsqueeze(3) * ins_LD_new

        x = torch.cat((ins_U_new, ins_D_new, ins_L_new, ins_R_new, ins_LU_new, ins_RD_new, ins_RU_new, ins_LD_new), 1)
        out = F.relu(self.transition_3(self.transition_3_bn(x)),inplace=True)

        return out


class encode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbnet_layer1 = nn.Sequential(*(list(model.children())[0][:4])) # 0~3 x 4
        self.mbnet_layer2 = nn.Sequential(*(list(model.children())[0][4:7])) # 4~6 x 2
        self.mbnet_layer3 = nn.Sequential(*(list(model.children())[0][7:13])) # 7~12 x 2
        self.mbnet_layer4 = nn.Sequential(*(list(model.children())[0][13:]),
                                *(list(model.children())[1])) # 0: (13~15) + 1: (*) x 2

    def forward(self, x):
        x1 = self.mbnet_layer1(x)
        x2 = self.mbnet_layer2(x1)
        x3 = self.mbnet_layer3(x2)
        x4 = self.mbnet_layer4(x3)
        return x1, x2, x3, x4 # x1 -> 24, x2 -> 40, x3 -> 112, x4 -> 960channels


class encode1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbnet = nn.Sequential(*list(model.children())[:2])

    def forward(self, x):
        x = self.mbnet(x)
        return x

class encode_small(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbnet_layer1 = nn.Sequential(*(list(model.children())[0][:2])) # 0~1 x 4
        self.mbnet_layer2 = nn.Sequential(*(list(model.children())[0][2:4])) # 2~3 x 2
        self.mbnet_layer3 = nn.Sequential(*(list(model.children())[0][4:9])) # 4~8 x 2
        self.mbnet_layer4 = nn.Sequential(*(list(model.children())[0][9:]),
                                *(list(model.children())[1])) # 0: (9~11) + 1: (*) x 2

    def forward(self, x):
        x1 = self.mbnet_layer1(x)
        x2 = self.mbnet_layer2(x1)
        x3 = self.mbnet_layer3(x2)
        x4 = self.mbnet_layer4(x3)
        return x1, x2, x3, x4


class encode_small1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mbnet = nn.Sequential(*list(model.children())[:2])

    def forward(self, x):
        x = self.mbnet(x)
        return x

##################### Final model assembly #####################
class OSnet_mb_frozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = encode(model=mobilenet_v3_module(pretrain=True, size='large',freeze=True))
        # self.encode1 = encode1(model=mobilenet_v3_module(pretrain=True, size='large',freeze=True))
        self.ca = ChannelAttention(in_planes=1024) # 1024
        self.sa = SpatialAttention()
        self.encode_texture = Deep_Orientation(960, 2048, 256) # 2048, 2048, 512
        # self.encode_texture1 = Deep_Orientation(960, 2048, 256) # 2048, 2048, 512
        self.embedding = nn.Sequential(
                nn.Conv2d(512, 1024, 1), # 512, 1024, 1
                nn.BatchNorm2d(1024), # 1024
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 1), # 1024, 1024, 1
                nn.BatchNorm2d(1024), # 1024
                nn.ReLU(inplace=True)
                )

        self.decode2 = nn.Sequential(
            nn.Conv2d(1136, 512, 1), #1136 # 2048, 512, 1
            nn.BatchNorm2d(512), # 512
            nn.ReLU(inplace=True),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(552, 256, 1), #552 # 1024, 256, 1
            nn.BatchNorm2d(256), # 256
            nn.ReLU(inplace=True),
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(280, 512, 1), #280 # 512, 512, 1
            nn.BatchNorm2d(512), # 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1), # 512, 1, 1
        )

        # self.ConvTrans1 = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 1024, 1, stride=2, output_padding=1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True)
        # )

        # self.ConvTrans2 = nn.Sequential(
        #     nn.ConvTranspose2d(1136, 512, 1, stride=2, output_padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        # self.ConvTrans3 = nn.Sequential(
        #     nn.ConvTranspose2d(552, 256, 1, stride=2, output_padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )

        # self.ConvTrans4 = nn.Sequential(
        #     nn.ConvTranspose2d(280, 1, 1, stride=4, output_padding=3),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, image, patch):
        img1, img2, img3, img4 = self.encode(image) # backbone encoding
        pat4 = self.encode(patch)[3]                  # backbone encoding  ######################## share module
        img4 = self.encode_texture(img4)            # Directionality-Aware Module
        pat4 = self.encode_texture(pat4)           # Directionality-Aware Module

        # Global Context Module
        for i in range(8):
            img_g = img4[:, 256 * i:256 * i + 256, :, :]
            pat = pat4[:, 256 * i:256 * i + 256, :, :]
            img_g = torch.cat([img_g, pat], dim=1)
            img_g = self.embedding(img_g)
            ca = self.ca(img_g) # Channel-Wise Attention
            img_g = ca * img_g  # Ablation study? 

            sa = self.sa(img_g) # Spatial-Wise Attention
            img_g = sa * img_g 

            if i == 0:
                img = img_g
            else:
                img += img_g
        
        # Decoder
        img = F.interpolate(img, 16, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img3], dim=1)
        img = self.decode2(img)
        img = F.interpolate(img, 32, mode='bilinear', align_corners=False) # bilinear upsampling x 2
        # img = self.ConvTrans2(img)

        img = torch.cat([img, img2], dim=1)
        img = self.decode3(img)
        img = F.interpolate(img, 64, mode='bilinear', align_corners=False) # bilinear upsampling x 2
        # img = self.ConvTrans3(img)

        img = torch.cat([img, img1], dim=1)
        img = self.decode4(img)
        img = F.interpolate(img, 256, mode='bilinear', align_corners=False) # bilinear upsampling x 4
        # img = self.ConvTrans4(img)

        img = torch.sigmoid(img)
        return img

class OSnet_mb_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = encode_small(model=mobilenet_v3_module(pretrain=True, size='small',freeze=True))
        self.encode1 = encode_small1(model=mobilenet_v3_module(pretrain=True, size='small',freeze=True))
        self.ca = ChannelAttention(in_planes=1024)
        # self.sa = SpatialAttention()
        self.encode_texture = Deep_Orientation(576, 2048, 512) # 2048, 2048, 512
        self.encode_texture1 = Deep_Orientation(576, 2048, 512) # 2048, 2048, 512
        self.embedding = nn.Sequential(
                nn.Conv2d(512, 1024, 1), # 512, 1024, 1
                nn.BatchNorm2d(1024), # 1024
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 1), # 1024, 1024, 1
                nn.BatchNorm2d(1024), # 1024
                nn.ReLU(inplace=True)
                )

        self.decode2 = nn.Sequential(
            nn.Conv2d(1072, 512, 1), # 2048, 512, 1
            nn.BatchNorm2d(512), # 512
            nn.ReLU(inplace=True),
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(536, 256, 1), # 1024, 256, 1
            nn.BatchNorm2d(256), # 256
            nn.ReLU(inplace=True),
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(272, 512, 1), # 512, 512, 1
            nn.BatchNorm2d(512), # 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1), # 512, 1, 1
        )

    def forward(self, image, patch):
        img1, img2, img3, img4 = self.encode(image) # backbone encoding
        pat4 = self.encode1(patch)                  # backbone encoding
        img4 = self.encode_texture(img4)            # Directionality-Aware Module
        pat4 = self.encode_texture1(pat4)           # Directionality-Aware Module

        # Global Context Module
        for i in range(8):
            img_g = img4[:, 256 * i:256 * i + 256, :, :]
            pat = pat4[:, 256 * i:256 * i + 256, :, :]
            img_g = torch.cat([img_g, pat], dim=1)
            img_g = self.embedding(img_g)
            ca = self.ca(img_g) # Channel-Wise Attention
            img_g = ca * img_g
            if i == 0:
                img = img_g
            else:
                img += img_g


        # Decoder
        img = F.interpolate(img, 16, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img3], dim=1)
        img = self.decode2(img)
        img = F.interpolate(img, 32, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img2], dim=1)
        img = self.decode3(img)
        img = F.interpolate(img, 64, mode='bilinear', align_corners=False) # bilinear upsampling x 2

        img = torch.cat([img, img1], dim=1)
        img = self.decode4(img)
        img = F.interpolate(img, 256, mode='bilinear', align_corners=False) # bilinear upsampling x 4

        img = torch.sigmoid(img)
        return img