import torch
import torch.nn as nn
import torch.nn.functional as F


class DRFE(nn.Module):
    def __init__(self, c1, num_recursions=3, reduction=16):
        super(DRFE, self).__init__()
        
        self.c1 = c1
        self.num_recursions = num_recursions
        
        self.rcfe = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(c1, max(2, c1 // reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, c1 // reduction), c1),
                nn.Sigmoid()
            ) for _ in range(num_recursions)
        ])
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.conv1x1 = nn.Conv2d(c1 * 2, c1 // reduction, 1)
        self.conv3x3 = nn.Conv2d(c1 // reduction, c1 // reduction, 3, padding=1)
        
        self.bn = nn.BatchNorm2d(c1 // reduction)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention_conv = nn.Conv2d(c1 // reduction, c1, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.feature_learning = nn.Sequential(
            nn.Conv2d(c1, c1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        rcfe_out = x
        for i in range(self.num_recursions):
            pool_out = self.avg_pool(rcfe_out).view(b, c)
            attention_weights = self.rcfe[i](pool_out).view(b, c, 1, 1)
            rcfe_out = rcfe_out * attention_weights
        
        avg_pooled = self.avg_pool(rcfe_out)
        max_pooled = self.max_pool(rcfe_out)
        
        pooled_concat = torch.cat([avg_pooled, max_pooled], dim=1)
        
        compressed = self.conv1x1(pooled_concat)
        compressed = self.conv3x3(compressed)
        compressed = self.bn(compressed)
        compressed = self.relu(compressed)
        
        attention_vector = self.attention_conv(compressed)
        attention_vector = self.sigmoid(attention_vector)
        
        weighted_features = rcfe_out * attention_vector
        
        fused_features = weighted_features + x
        
        output = self.feature_learning(fused_features)
        
        return output


if __name__ == "__main__":
    x = torch.randn(2, 1024, 32, 32)
    drfe = DRFE(c1=1024, num_recursions=2, reduction=16)
    
    output = drfe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in drfe.parameters()):,}")
    
    x2 = torch.randn(2, 256, 64, 64)
    drfe2 = DRFE(c1=256, num_recursions=2, reduction=16)
    output2 = drfe2(x2)
    print(f"\nTest 2 - Input shape: {x2.shape}")
    print(f"Test 2 - Output shape: {output2.shape}")