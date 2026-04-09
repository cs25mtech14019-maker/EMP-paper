import torch
import torch.nn as nn


class LaneEmbeddingLayer(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(feat_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1),
        )

    def forward(self, x, mask=None):
        bs, n, _ = x.shape

        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        
        if mask is not None:
            feature_pool = feature.masked_fill(mask.unsqueeze(1), float('-inf'))
        else:
            feature_pool = feature
        feature_global = torch.max(feature_pool, dim=2, keepdim=True)[0]  # B 256 1
        feature_global.masked_fill_(feature_global == float('-inf'), 0.0)
            
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # B 512 n
        
        feature = self.second_conv(feature)  # B c n
        
        if mask is not None:
            feature_pool2 = feature.masked_fill(mask.unsqueeze(1), float('-inf'))
        else:
            feature_pool2 = feature
        feature_global2 = torch.max(feature_pool2, dim=2, keepdim=False)[0]  # B c
        feature_global2.masked_fill_(feature_global2 == float('-inf'), 0.0)
            
        return feature_global2
