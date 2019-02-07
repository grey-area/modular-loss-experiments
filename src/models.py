import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging


activations = {
    'relu': F.relu,
    'sigmoid': F.sigmoid
}


class ModelMixin:
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


class DebugTerminateMixin:
    def debug_terminate(self):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            sys.exit()


class ModularLinear(nn.Module, ModelMixin):

    def __init__(self, in_features, out_features,
                 num_modules, bias=True):
        super(ModularLinear, self).__init__()

        if bias is not True:
            raise RuntimeError('Only supported with bias for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules

        # If dimension of weights is M x In x Out
        #  Then M will broadcast if incoming matrix
        #  is N x In, giving M x N x Out
        self.weight = nn.Parameter(torch.Tensor(
            self.num_modules, self.in_features, self.out_features
        ))
        self.bias = nn.Parameter(torch.Tensor(
            self.num_modules, 1, self.out_features
        ))
        self.reset_parameters()

    def reset_parameters(self):
        in_features = self.weight.size(1)
        std_dev = 1. / math.sqrt(in_features)
        self.weight.data.uniform_(-std_dev, std_dev)
        self.bias.data.uniform_(-std_dev, std_dev)

    def forward(self, x):
        # Broadcasts if input is a lower dimensional tensor,
        #  as in the case of the initial design matrix.
        #  In future layers, this should do a batch
        #  matrix multiply.
        return torch.matmul(x, self.weight) + self.bias


class DenseModel(nn.Module, ModelMixin, DebugTerminateMixin):
    def __init__(self, input_dim=1, output_dim=1, hidden_nodes=5,
                 num_modules=1, activation='relu'):
        super(DenseModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.num_modules = num_modules

        layers = []
        nodes = [input_dim] + hidden_nodes + [output_dim]

        for n_i in range(len(nodes))[:-1]:
            layers.append(ModularLinear(nodes[n_i],
                                        nodes[n_i + 1],
                                        num_modules))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        logging.debug('Shapes per layer:')

        x = x.view(-1, self.input_dim)
        logging.debug(x.shape)

        for layer in self.layers[:-1]:
            x = activations[self.activation](layer(x))
            logging.debug(x.shape)

        x = self.layers[-1](x)
        logging.debug(f'{x.shape}\n')

        self.debug_terminate()

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def weight_grad_norms(self):

        weight_norms = np.zeros((self.num_modules, len(self.layers), 2))
        grad_norms = np.zeros((self.num_modules, len(self.layers), 2))

        for layer_i, layer in enumerate(self.layers):
            weight_norms[:, layer_i, 0] = torch.sqrt(torch.sum(layer.weight.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            weight_norms[:, layer_i, 1] = torch.sqrt(torch.sum(layer.bias.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            grad_norms[:, layer_i, 0] = torch.sqrt(torch.sum(layer.weight.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            grad_norms[:, layer_i, 1] = torch.sqrt(torch.sum(layer.bias.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()

        return np.expand_dims(weight_norms, axis=0), np.expand_dims(grad_norms, axis=0)


class ConvModel(nn.Module, ModelMixin, DebugTerminateMixin):
    def __init__(self, input_shape, output_dim, num_filters,
                 kernel_sizes, strides, dilations, num_modules,
                 activation, final_layer):
        super(ConvModel, self).__init__()

        self.input_channels = input_shape[0]
        current_shape = np.array(input_shape[1:])
        self.output_dim = output_dim
        self.num_modules = num_modules
        self.num_filters = num_filters
        if len(kernel_sizes) == 1:
            kernel_sizes *= len(num_filters)
        if len(strides) == 1:
            strides *= len(num_filters)
        if len(dilations) == 1:
            dilations *= len(num_filters)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilations = dilations
        self.activation = activation
        self.final_layer = final_layer

        self.filters_list = [self.input_channels] + self.num_filters

        layers = []
        for f_i, f_o, k, s, d in zip(self.filters_list[:-1],
                                     self.filters_list[1:],
                                     kernel_sizes, strides, dilations):
            layers.append(
                nn.Conv2d(
                    num_modules * f_i, num_modules * f_o, (k, k),
                    groups=num_modules, bias=True, stride=s, dilation=d
                )
            )

            current_shape = np.floor((current_shape - d * (k - 1) - 1) / s + 1)
        current_shape = current_shape.astype(np.int32)

        self.layers = nn.ModuleList(layers)

        if self.final_layer == 'avg': # average pooling before fc
            self.fc = ModularLinear(self.filters_list[-1], output_dim, num_modules)
        else: # fc
            self.fc = ModularLinear(self.filters_list[-1] * np.prod(current_shape), output_dim, num_modules)


    def forward(self, x):
        logging.debug('Shapes per layer:')

        x = x.repeat(1, self.num_modules, 1, 1)
        n_examples = x.shape[0]

        logging.debug(x.shape)

        for layer in self.layers:
            x = activations[self.activation](layer(x))
            logging.debug(x.shape)

        if self.final_layer == 'avg':
            kernel = x.shape[2:]
            remaining_shape = x.shape[:2]
            pool = nn.AvgPool2d(kernel)
            x = pool(x)
            logging.debug(x.shape)

        # Turning into 3D tensor M x N x K
        # N examples, M modules, K classes / features
        conv2_3D = x.view(
            n_examples, self.num_modules, -1
        ).permute(1, 0, 2)
        logging.debug(conv2_3D.shape)

        x = self.fc(conv2_3D)
        logging.debug(f'{x.shape}\n')

        self.debug_terminate()

        return x

    def weight_grad_norms(self):
        weight_norms = np.zeros((self.num_modules, len(self.layers) + 1, 2))
        grad_norms = np.zeros((self.num_modules, len(self.layers) + 1, 2))

        for layer_i, layer in enumerate(self.layers):
            weight_norms[:, layer_i, 0] = torch.sqrt(torch.sum(layer.weight.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            weight_norms[:, layer_i, 1] = torch.sqrt(torch.sum(layer.bias.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            grad_norms[:, layer_i, 0] = torch.sqrt(torch.sum(layer.weight.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
            grad_norms[:, layer_i, 1] = torch.sqrt(torch.sum(layer.bias.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()

        weight_norms[:, -1, 0] = torch.sqrt(torch.sum(self.fc.weight.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
        weight_norms[:, -1, 1] = torch.sqrt(torch.sum(self.fc.bias.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
        grad_norms[:, -1, 0] = torch.sqrt(torch.sum(self.fc.weight.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()
        grad_norms[:, -1, 1] = torch.sqrt(torch.sum(self.fc.bias.grad.data.reshape(self.num_modules, -1)**2, dim=1)).cpu().numpy()

        return np.expand_dims(weight_norms, axis=0), np.expand_dims(grad_norms, axis=0)

    def reset_parameters(self):
        for in_filters, k, layer in zip(self.filters_list[:-1], self.kernel_sizes, self.layers):
            std_dev = 1. / (math.sqrt(k * k * in_filters))
            layer.weight.data.uniform_(-std_dev, std_dev)
            layer.bias.data.uniform_(-std_dev, std_dev)

        self.fc.reset_parameters()


class Bottleneck(nn.Module):
    def __init__(self, n_channels, growth_rate, num_modules):
        super(Bottleneck, self).__init__()

        self.num_modules = num_modules

        interChannels = 4*growth_rate
        self.bn1 = nn.BatchNorm2d(num_modules * n_channels)
        self.conv1 = nn.Conv2d(num_modules * n_channels, num_modules * interChannels, kernel_size=1,
                               bias=False, groups=num_modules)
        self.bn2 = nn.BatchNorm2d(num_modules * interChannels)
        self.conv2 = nn.Conv2d(num_modules * interChannels, num_modules * growth_rate, kernel_size=3,
                               padding=1, bias=False, groups=num_modules)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))

        B, F1, S, _ = x.shape
        F2 = out.shape[1]
        x = x.view(B, self.num_modules, F1//self.num_modules, S, S)
        out = out.view(B, self.num_modules, F2//self.num_modules, S, S)

        out = torch.cat((x, out), 2).view(B, F1+F2, S, S)
        return out


class SingleLayer(nn.Module):
    def __init__(self, n_channels, growth_rate, num_modules):
        super(SingleLayer, self).__init__()

        self.num_modules = num_modules

        self.bn1 = nn.BatchNorm2d(num_modules * n_channels)
        self.conv1 = nn.Conv2d(num_modules * n_channels, num_modules * growth_rate, kernel_size=3,
                               padding=1, bias=False, groups=num_modules)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))

        B, F1, S, _ = x.shape
        F2 = out.shape[1]
        x = x.view(B, self.num_modules, F1//self.num_modules, S, S)
        out = out.view(B, self.num_modules, F2//self.num_modules, S, S)

        out = torch.cat((x, out), 2).view(B, F1+F2, S, S)
        return out

class Transition(nn.Module):
    def __init__(self, n_channels, n_out_channels, num_modules):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_modules * n_channels)
        self.conv1 = nn.Conv2d(num_modules * n_channels, num_modules * n_out_channels, kernel_size=1,
                               bias=False, groups=num_modules)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module, ModelMixin, DebugTerminateMixin):
    def __init__(self, input_shape, output_dim, growth_rate, depth, reduction, bottleneck, num_modules=1):
        super(DenseNet, self).__init__()

        self.output_dim = output_dim
        self.num_modules = num_modules
        input_channels = input_shape[0]

        n_dense_blocks = (depth-4) // 3
        if bottleneck:
            n_dense_blocks //= 2

        n_channels = 2*growth_rate
        self.conv1 = nn.Conv2d(num_modules * input_channels, num_modules * n_channels,
                               kernel_size=3, padding=1, bias=False, groups=num_modules)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, num_modules)
        n_channels += n_dense_blocks*growth_rate
        n_out_channels = int(math.floor(n_channels*reduction))
        self.trans1 = Transition(n_channels, n_out_channels, num_modules)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, num_modules)
        n_channels += n_dense_blocks*growth_rate
        n_out_channels = int(math.floor(n_channels*reduction))
        self.trans2 = Transition(n_channels, n_out_channels, num_modules)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, num_modules)
        n_channels += n_dense_blocks*growth_rate

        self.bn1 = nn.BatchNorm2d(num_modules * n_channels)
        self.fc = ModularLinear(n_channels, output_dim, num_modules)

        self.reset_parameters()

    def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck, num_modules):
        layers = []
        for i in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(Bottleneck(n_channels, growth_rate, num_modules))
            else:
                layers.append(SingleLayer(n_channels, growth_rate, num_modules))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        logging.debug('Shapes per layer:')

        out = x.repeat(1, self.num_modules, 1, 1)
        n_examples = out.shape[0]
        logging.debug(out.shape)

        for layer in [self.conv1, self.dense1, self.trans1, self.dense2,
                      self.trans2, self.dense3]:
            out = layer(out)
            logging.debug(out.shape)

        out = torch.squeeze(F.avg_pool2d(torch.relu(self.bn1(out)), 8))
        logging.debug(out.shape)

        out = out.view(n_examples, self.num_modules, -1).permute(1, 0, 2)
        out = self.fc(out)
        logging.debug(f'{out.shape}\n')

        self.debug_terminate()

        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / self.num_modules
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, ModularLinear):
                m.reset_parameters()
                m.bias.data.zero_()

    def weight_grad_norms(self):
        # Not implemented
        return np.zeros(10), np.zeros(10)
