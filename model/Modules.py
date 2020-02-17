"""Includes all basic modules for building the network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvBlock, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        out = self.conv(x)
        return out

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class Pyramid(nn.Module):
    def __init__(self, pyr_in_channel, pyr_out_channel, kernel_sizes, strides, paddings):
        super(Pyramid, self).__init__()
        
        convolutions = []
        num_layers = len(kernel_sizes)
        for i in range(num_layers):
            conv_layer = nn.Sequential(
                ConvBlock(pyr_in_channel if i == 0 else pyr_out_channel,
                         pyr_out_channel,
                         kernel_size=kernel_sizes[i], stride=strides[i],
                         padding=paddings[i],
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(pyr_out_channel)) if i != num_layers-1 else nn.Sequential(
                ConvBlock(pyr_in_channel if i == 0 else pyr_out_channel,
                         pyr_out_channel,
                         kernel_size=kernel_sizes[i], stride=strides[i],
                         padding=paddings[i],
                         dilation=1, w_init_gain='relu'))
            convolutions.append(conv_layer)
        
        self.convolutions = nn.ModuleList(convolutions)
        
    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        
        return x

class Extractor(nn.Module):
    def __init__(self, inc, lac):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv1d(inc, lac, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(inc+lac, lac, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self,x):
        out = torch.cat((self.conv1(x), x),1)
        out = torch.cat((self.conv2(out), out),1)
        
        return out

class FeatureCorrelation(torch.nn.Module):
    def __init__(self,normalization=False):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        feature_B = feature_B.transpose(1,2)
        correlation_tensor = torch.bmm(feature_B,feature_A)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        # (batch, audio_dim, video_dim)
        return correlation_tensor
