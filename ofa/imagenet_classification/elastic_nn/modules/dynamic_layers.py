# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn
from collections import OrderedDict

from ofa.utils.layers import MBConvLayer, ConvLayer, IdentityLayer, set_layer_from_config
from ofa.utils.layers import ResNetBottleneckBlock, LinearLayer, MultiHeadLinearLayer, ResNetResidualBlock
from ofa.utils import MyModule, val2list, get_net_device, build_activation, make_divisible, SEModule, MyNetwork
from .dynamic_op import DynamicMaskConv2d,DynamicSeparableConv2d, DynamicConv2d, DynamicBatchNorm2d, DynamicSE, DynamicGroupNorm
from .dynamic_op import DynamicLinear,DynamicMaskLinear

__all__ = [
    'adjust_bn_according_to_idx', 'copy_bn','DynamicMaskResidualBlock','DynamicMaskConvLayer','DynamicMaskLinearLayer'
    'DynamicMBConvLayer', 'DynamicConvLayer', 'DynamicLinearLayer', 'DynamicResNetBottleneckBlock'
]


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_channels if isinstance(target_bn, nn.GroupNorm) else target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


class DynamicLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features, bias=self.bias
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return 'DyLinear(%d, %d)' % (max(self.in_features_list), self.out_features)

    @property
    def config(self):
        return {
            'name': DynamicLinear.__name__,
            'in_features_list': self.in_features_list,
            'out_features': self.out_features,
            'bias': self.bias,
            'dropout_rate': self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.get_active_bias(self.out_features).data
            )
        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            'name': LinearLayer.__name__,
            'in_features': in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'dropout_rate': self.dropout_rate,
        }

class DynamicMultiHeadLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, num_heads=1, bias=True, dropout_rate=0):
        super(DynamicMultiHeadLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.num_heads = num_heads

        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        
        self.layers = nn.ModuleList()
        for k in range(num_heads):
            layer = DynamicLinear(
            max_in_features=max(self.in_features_list),
                max_out_features=self.out_features,
                bias=self.bias
            )
            self.layers.append(layer)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        output_list = []
        for layer in self.layers:
            output = layer.forward(x)
            output_list.append(output)

        outputs = torch.cat(tuple(out[:,:-1] for out in output_list),1)
        return outputs, output_list

    @property
    def module_str(self):
        return 'DyLinear(%d, %d)' % (max(self.in_features_list), self.out_features)

    @property
    def config(self):
        return {
            'name': DynamicLinear.__name__,
            'in_features_list': self.in_features_list,
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'bias': self.bias,
            'dropout_rate': self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMultiHeadLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = MultiHeadLinearLayer(in_features, self.out_features, self.num_heads, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # Modified to be compatible with ensemble model
        for l in range(len(sub_layer.layers)):
            sub_layer.layers[l].weight.data.copy_( 
                self.layers[l].get_active_weight(self.out_features, in_features).data
            )
            if self.bias:
                sub_layer.layers[l].bias.data.copy_(
                    self.layers[l].get_active_bias(self.out_features).data
                )
        return sub_layer
    
    def get_active_branches(self, idx, in_features, preserve_weight=True):
        sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_( 
            self.layers[idx].get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.layers[idx].get_active_bias(self.out_features).data
            )
        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            'name': LinearLayer.__name__,
            'in_features': in_features,
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'bias': self.bias,
            'dropout_rate': self.dropout_rate,
        }

class DynamicMBConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, self.stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), MyNetwork.CHANNEL_DIVISIBLE)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)

    @property
    def config(self):
        return {
            'name': DynamicMBConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(round(in_channel * self.active_expand_ratio), MyNetwork.CHANNEL_DIVISIBLE)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)
        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.get_active_filter(middle_channel, in_channel).data,
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=MyNetwork.CHANNEL_DIVISIBLE)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.get_active_reduce_bias(se_mid).data
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.get_active_expand_bias(middle_channel).data
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.get_active_filter(self.active_out_channel, middle_channel).data
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': MBConvLayer.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.active_kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channel(in_channel),
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.depth_conv.bn, DynamicGroupNorm):
            channel_per_group = self.depth_conv.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.in_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]

            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx

class DynamicConvLayer(MyModule):

    def __init__(self, ens, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6'):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        self.ens = ens

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func)

        self.active_out_channel = max(self.out_channel_list)

        a,b,c,d = self.conv.shape
        self.prune_mask = torch.ones(self.ens, a, b, c, d)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            'name': DynamicConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.get_active_filter(self.active_out_channel, in_channel).data)
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer
    
    def get_active_subnet_config(self, in_channel):
        return {
            'name': ConvLayer.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }

    # def set_ensemble(self, ens, in_channel):

class DynamicResNetBottleneckBlock(MyModule):

    def __init__(self, in_channel_list, out_channel_list, expand_ratio_list=0.25,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='avgpool_conv'):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, kernel_size, stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list))),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_BottleneckConv_in->%d->%d_S%d' % (
                self.kernel_size, self.kernel_size, self.active_middle_channels, self.active_out_channel, self.stride
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': DynamicResNetBottleneckBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'expand_ratio_list': self.expand_ratio_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'downsample_mode': self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResNetBottleneckBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(self.active_middle_channels, in_channel).data)
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(self.active_middle_channels, self.active_middle_channels).data)
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        sub_layer.conv3.conv.weight.data.copy_(
            self.conv3.conv.get_active_filter(self.active_out_channel, self.active_middle_channels).data)
        copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(self.active_out_channel, in_channel).data)
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': ResNetBottleneckBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channels,
            'act_func': self.act_func,
            'groups': 1,
            'downsample_mode': self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # conv3 -> conv2
        importance = torch.sum(torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.conv2.bn, DynamicGroupNorm):
            channel_per_group = self.conv2.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv3.conv.conv.weight.data = torch.index_select(self.conv3.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 0, sorted_idx)

        # conv2 -> conv1
        importance = torch.sum(torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.conv1.bn, DynamicGroupNorm):
            channel_per_group = self.conv1.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(self.conv1.conv.conv.weight.data, 0, sorted_idx)

        return None

class DynamicResNetResidualBlock(MyModule):

    def __init__(self, ens, in_channel_list, out_channel_list, expand_ratio_list=1,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='conv'):
        super(DynamicResNetResidualBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode
        self.ens = ens

        

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size, stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list))),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)


    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_BottleneckConv_in->%d->%d_S%d' % (
                self.kernel_size, self.kernel_size, self.active_middle_channels, self.active_out_channel, self.stride
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': DynamicResNetResidualBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'expand_ratio_list': self.expand_ratio_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'downsample_mode': self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResNetResidualBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(self.active_middle_channels, in_channel).data)
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        # sub_layer.conv2.conv.weight.data.copy_(
        #     self.conv2.conv.get_active_filter(self.active_middle_channels, self.active_middle_channels).data)
        # copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(self.active_out_channel, self.active_middle_channels).data)
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(self.active_out_channel, in_channel).data)
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': ResNetResidualBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channels,
            'act_func': self.act_func,
            'groups': 1,
            'downsample_mode': self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # conv2 -> conv1
        importance = torch.sum(torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3))
        # import pdb; pdb.set_trace()
        if isinstance(self.conv1.bn, DynamicGroupNorm):
            channel_per_group = self.conv1.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(self.conv1.conv.conv.weight.data, 0, sorted_idx)

        return None
    
    def re_organize_middle_weights_v2(self,expand_ratio_stage=0):
        # conv2 -> conv1
        conv2_ori_weight = copy.deepcopy(self.conv2.conv.conv.weight.data)
        conv2_ori_weight_grad = copy.deepcopy(self.conv2.conv.conv.weight.grad)

        conv1_ori_weight = copy.deepcopy(self.conv1.conv.conv.weight.data)
        # conv1_ori_weight_grad = copy.deepcopy(self.conv1.conv.conv.weight.grad)

        importance = torch.sum(conv2_ori_weight * conv2_ori_weight_grad, dim=(0, 2, 3))
        importance *= importance
        # import pdb; pdb.set_trace()
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(conv2_ori_weight, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(conv1_ori_weight, 0, sorted_idx)

        return None

################################################
class DynamicMaskConvLayer(MyModule):
    '''
    与DynamicConvLayer 区别： 1.调用的是 DynamicMaskConv2d; 2.有n个branches
    warning: 类下其它方法对多branches适配未验证
    '''
    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6',branches=2):
        super(DynamicMaskConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        self.branches = branches

        self.conv = DynamicMaskConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, branches=branches
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func)

        self.active_out_channel = max(self.out_channel_list)
    
    def compute_mask(self, idx,pruning_rate=0):
        self.conv.compute_mask(idx, pruning_rate)

    def forward(self, x, idx=0.2):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x,idx)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            'name': DynamicMaskConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'branches': self.branches
        }

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)


class DynamicMaskResidualBlock(MyModule):

    def __init__(self, ens, in_channel_list, out_channel_list, expand_ratio_list=1,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='conv', branches=2):
        super(DynamicMaskResidualBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode
        self.ens = ens

        

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)

        self.conv1 = DynamicMaskConv2d(
            max_in_channels = max(self.in_channel_list),
            max_out_channels = max_middle_channel,
            kernel_size=kernel_size,
            stride=stride,
            branches=branches,
            )
        self.bn1 = DynamicBatchNorm2d(max_middle_channel)
        self.act =  build_activation(self.act_func, inplace=True)


        self.conv2 = DynamicMaskConv2d(
            max_in_channels = max_middle_channel,
            max_out_channels = max(self.out_channel_list),
            kernel_size = kernel_size,
            branches=branches,
            )
        self.bn2 = DynamicBatchNorm2d(max(self.out_channel_list))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.res = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.res = DynamicMaskConv2d(
                max_in_channels = max(self.in_channel_list),
                max_out_channels = max(self.out_channel_list),
                stride=stride,
                branches=branches,
                )
            self.bn_res = ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)


    def compute_mask(self, idx, pruning_rate_list=[0.2,0.2,0.2]):
        self.conv1.compute_mask(idx, pruning_rate_list[0])
        self.conv2.compute_mask(idx, pruning_rate_list[1])  
        self.res.compute_mask(idx, pruning_rate_list[2])

    def forward(self, x, idx):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.res, IdentityLayer):
            self.res.conv.active_out_channel = self.active_out_channel

        residual = self.res(x, idx)

        x = self.conv1(x, idx)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x, idx)
        x = self.bn2(x)

        x = x + residual
        x = self.final_act(x)
        return x

class DynamicMaskLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0,branches=2):
        super(DynamicMaskLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicMaskLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features, bias=self.bias, branches=branches
        )
    
    def compute_mask(self, idx, pruning_rate=0.5):
        self.linear.compute_mask(idx, pruning_rate)

    def forward(self, x, idx):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x, idx)