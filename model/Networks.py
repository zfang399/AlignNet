# Finished by zfang 2020/02/15 18:15pm
import random
import torch
import torch.nn as nn
from model.Modules import Pyramid, Extractor, FeatureCorrelation
import torch.nn.functional as F

class AlignNet(nn.Module):
    def __init__(self, cfg):
        super(AlignNet, self).__init__()
        self.cfg = cfg
        ################################################################################
        # 0. Attention
        ################################################################################
        self.kp_att = torch.nn.Parameter(torch.ones(self.cfg["num_lip_attention"]),requires_grad = True)
        torch.nn.init.uniform(self.kp_att.data, -0.005, 0.005)
        self.softmax = nn.Softmax(dim = -1)

        ################################################################################
        # 1. Pyramids (Forward Part)
        ################################################################################
        # video
        video_pyrs = []
        for i in range(self.cfg["num_pyrs"]):
            video_pyrs.append(Pyramid(pyr_in_channel = self.cfg["num_lip_keypoints"]*4, 
                                       pyr_out_channel = self.cfg["pyr_channel"],
                                       kernel_sizes = [5,3,3,3,3],
                                       strides = [3,1,1,1,1],
                                       paddings = [1,1,1,1,1]) if i == 0 else Pyramid(
                                       pyr_in_channel = self.cfg["pyr_channel"] // (2**(i-1)),
                                       pyr_out_channel = self.cfg["pyr_channel"] // (2**i),
                                       kernel_sizes = [3,3,3],
                                       strides = [2,1,1],
                                       paddings = [1,1,1]))
        self.video_pyrs = nn.ModuleList(video_pyrs)
        # audio
        audio_pyrs = []
        for i in range(self.cfg["num_pyrs"]):
            audio_pyrs.append(Pyramid(pyr_in_channel = self.cfg["num_mels"], 
                                       pyr_out_channel = self.cfg["pyr_channel"],
                                       kernel_sizes = [7,5,5,3,3],
                                       strides = [4,2,2,2,1],
                                       paddings = [4,1,1,1,1]) if i == 0 else Pyramid(
                                       pyr_in_channel = self.cfg["pyr_channel"] // (2**(i-1)),
                                       pyr_out_channel = self.cfg["pyr_channel"] // (2**i),
                                       kernel_sizes = [3,3,3],
                                       strides = [2,1,1],
                                       paddings = [1,1,1]))
        self.audio_pyrs = nn.ModuleList(audio_pyrs)

        ################################################################################
        # 2. Correlation, Extraction, Predictions (Backward Part)
        ################################################################################
        # correlation
        self.correlation = FeatureCorrelation()

        # feature extractors & flow prediction layers
        self.base_corr_channel = 64 # fixed for 12 second audio, will have to change the structure if otherwise
        self.base_add_channel = self.cfg["addon_channel"]
        self.base_pyr_channel = self.cfg["pyr_channel"]
        extractors = []
        predictors = []
        for i in range(self.cfg["num_pyrs"]):
            # the last layer does not have the extra dimension for flow predicted from detailed layer
            if i == self.cfg["num_pyrs"]-1:
                extractor_channel_in = int(self.base_corr_channel/(2**i) + self.base_pyr_channel/(2**i))
            else:
                extractor_channel_in = int(self.base_corr_channel/(2**i) + self.base_pyr_channel/(2**i)) + 1
            extractor_channel_add = int(self.base_add_channel/(2**i))
            extractors.append(Extractor(extractor_channel_in, extractor_channel_add))

            predictor_channel_in = extractor_channel_in + 2*extractor_channel_add
            predictors.append(nn.Conv1d(predictor_channel_in, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.extractors = nn.ModuleList(extractors)
        self.predictors = nn.ModuleList(predictors)

        # direct predictor
        dp_channel_in = self.base_corr_channel + self.base_pyr_channel + 1 + 2*self.base_add_channel
        dp_channel_out = self.cfg["dp_channel"]
        dp = []
        i = 0
        while dp_channel_in > 1:
            dp.append(nn.Conv1d(dp_channel_in, dp_channel_out, kernel_size=3, stride=1, padding=2**i, dilation=2**i, bias=True))
            dp_channel_in = dp_channel_out
            dp_channel_out //= 4
            i += 1
        self.dp = nn.ModuleList(dp)
        
    def warp(self, x, indices):
        '''
            x: [B, C, L] (feature) --- batch * channel * length
            indices: [B, 1, L] indices    --- batch * 1 * length
        '''
        # indices are on the scale of (-1,1)
        B, C, L = x.size()
        x = x.unsqueeze(2)

        grid = torch.zeros(B, 1, L, 2)
        grid[:,:,:,0] = indices
        grid = grid.cuda()

        output = nn.functional.grid_sample(x, grid, align_corners=True)

        ret = output
        ret = ret.view(B,C,L)

        return ret
        
    def forward(self, video_feature, audio_feature):
        ################################################################################
        # 0. Attention & Reshaping
        ################################################################################
        B, L0, _, _, _  = video_feature.size()
        # multiply keypoint attention
        sm_att = self.softmax(self.kp_att)
        att = sm_att[self.cfg["lip_attention_to_keypoints"]]
        att = att.view(1,1,self.cfg["num_lip_keypoints"],1,1).expand(B,L0,-1,2,2)
        video_feature *= att
        # reshape video feature
        video_feature = video_feature.reshape(B,L0,self.cfg["num_lip_keypoints"]*4)
        video_feature = video_feature.transpose(1,2)

        ################################################################################
        # 1. Forward Calculation (pyramids)
        ################################################################################
        video_pyr_features = []
        video_pyr_input = video_feature
        video_deconv_layers = [torch.nn.Upsample(L0, mode='linear')]
        for pyr in self.video_pyrs:
            video_pyr_input = pyr(video_pyr_input)
            video_pyr_features.append(video_pyr_input)
            video_deconv_layers.append(torch.nn.Upsample(video_pyr_input.size(2), mode='linear'))
        video_deconv_layers.pop()
        self.video_deconv_layers = nn.ModuleList(video_deconv_layers)

        audio_pyr_features = []
        audio_pyr_input = audio_feature
        audio_deconv_layers = []
        for pyr in self.audio_pyrs:
            audio_pyr_input = pyr(audio_pyr_input)
            audio_pyr_features.append(audio_pyr_input)
            audio_deconv_layers.append(torch.nn.Upsample(audio_pyr_input.size(2), mode='linear'))
        audio_deconv_layers.pop()
        self.audio_deconv_layers = nn.ModuleList(audio_deconv_layers)

        ################################################################################
        # 2. Back Calculation (extract, predict, warp, upsampling)
        ################################################################################
        outs = []
        up_flow = -1
        for i in range(self.cfg["num_pyrs"]-1,-1,-1):
            if i != self.cfg["num_pyrs"]-1:
                outs.append(up_flow.view(B, -1))
                # warp features if this is not the last pyramid
                up_flow_audio = self.audio_deconv_layers[i](flow)
                feature_warped = self.warp(audio_pyr_features[i], up_flow_audio)                
                corr = self.correlation(video_pyr_features[i], feature_warped)
                feature_cat = self.extractors[i](torch.cat((corr, video_pyr_features[i], up_flow), 1))
            else:
                corr = self.correlation(video_pyr_features[i], audio_pyr_features[i])
                feature_cat = self.extractors[i](torch.cat((corr, video_pyr_features[i]), 1))
            
            flow = self.predictors[i](feature_cat)
            up_flow = self.video_deconv_layers[i](flow)

        dp_flow = feature_cat
        for dp_conv in self.dp:
            dp_flow = F.relu(dp_conv(dp_flow))
            
        flow = flow + dp_flow
        up_flow = self.video_deconv_layers[0](flow)
        outs.append(up_flow.view(B, -1))
        outs.reverse()
        
        return outs
