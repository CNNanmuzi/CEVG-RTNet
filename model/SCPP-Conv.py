import torch
import torch.nn as nn
import torch.nn.functional as F

class SCEM(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(SCEM, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels * scale * scale, in_channels, 
                             kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        x = x.view(batch_size, channels, 
                   height // self.scale, self.scale,
                   width // self.scale, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * self.scale * self.scale,
                   height // self.scale, width // self.scale)
        
        x = self.conv(x)
        x = self.bn(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map

class SCPP_Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super(SCPP_Conv, self).__init__()
        self.c = c2
        self.k = k
        self.s = s
        p = p or k // 2
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(c1 * 2, c1 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // 4, c1),
            nn.Sigmoid()
        )
        
        self.depthwise_conv = nn.Conv2d(c1, c1, 
                                       kernel_size=k, 
                                       stride=s, 
                                       padding=p, 
                                       groups=c1, 
                                       bias=False)
        
        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.spatial_attention = SpatialAttention(c1)
        self.scem = SCEM(c1)
        
        self.output_conv = nn.Conv2d(c1, c2, 
                                   kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()
        
    def forward(self, x):
        identity = x
        batch_size, c1, height, width = x.size()
        
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        
        pooled_features = torch.cat([max_pooled, avg_pooled], dim=1)
        pooled_features = pooled_features.view(batch_size, -1)
        
        channel_prior = self.mlp(pooled_features)
        channel_prior = channel_prior.view(batch_size, c1, 1, 1)
        
        x_channel_mixed = x * channel_prior
        
        x_depth = self.depthwise_conv(x_channel_mixed)
        x_depth = self.bn1(x_depth)
        
        x_normalized = self.bn2(x_depth)
        x_spatial_att = self.spatial_attention(x_normalized)
        x_enhanced = self.scem(x_spatial_att)
        
        if self.s == 1 and identity.shape == x_enhanced.shape:
            x_out = x_enhanced + identity
        else:
            x_out = x_enhanced
        
        x_out = self.output_conv(x_out)
        x_out = self.bn_out(x_out)
        x_out = self.relu(x_out)
        
        return x_out

if __name__ == "__main__":
    scpp_conv = SCPP_Conv(c1=64, c2=128, k=3, s=2)
    x = torch.randn(2, 64, 32, 32)
    output = scpp_conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in scpp_conv.parameters()):,}")