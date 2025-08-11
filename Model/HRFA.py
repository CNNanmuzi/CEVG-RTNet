import torch
import torch.nn as nn
import torch.nn.functional as F

class ERFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERFB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out + residual)
        return out

class VGCABlock(nn.Module):
    def __init__(self, in_channels):
        super(VGCABlock, self).__init__()
        self.in_channels = in_channels
        self.graph_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.adjacency_learner = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        adjacency_weights = torch.sigmoid(self.adjacency_learner(x))
        graph_features = self.graph_conv(x * adjacency_weights)
        return graph_features

class DepthAwareGating(nn.Module):
    def __init__(self, in_channels, reduction_ratio=0.25):
        super(DepthAwareGating, self).__init__()
        reduced_channels = max(1, int(in_channels * reduction_ratio))
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gate = self.mlp(x)
        return x * gate

class HRFA(nn.Module):
    def __init__(self, c1, c2, enhanced_mode=False, reduction_ratio=0.25):
        super(HRFA, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.enhanced_mode = enhanced_mode
        
        self.initial_enhance = nn.Sequential(
            nn.Conv2d(c1, c2, 1),
            nn.ReLU(inplace=True)
        )
        
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, c2 * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 * 2, c2, 1)
        )
        
        self.erfb = ERFB(c2, c2)
        self.depth_gating = DepthAwareGating(c2, reduction_ratio)
        
        if enhanced_mode:
            self.vgca = VGCABlock(c2)
            self.ffn = nn.Sequential(
                nn.Conv2d(c2, c2 * 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2 * 4, c2, 1)
            )
        else:
            self.vgca = None
            self.ffn = None
        
    def forward(self, x):
        enhanced = self.initial_enhance(x)
        mlp_out = self.mlp(enhanced)
        erfb_out = self.erfb(mlp_out)
        gated_out = self.depth_gating(erfb_out)
        
        if self.enhanced_mode and self.vgca is not None:
            aligned_features = self.vgca(gated_out)
            final_output = self.ffn(aligned_features)
        else:
            final_output = gated_out
            
        return final_output

if __name__ == "__main__":
    print("Testing HRFA module with different configurations...")
    
    # Test 1: Config [256, False, 0.25]
    x1 = torch.randn(2, 128, 64, 64)
    hrfa1 = HRFA(c1=128, c2=256, enhanced_mode=False, reduction_ratio=0.25)
    output1 = hrfa1(x1)
    print(f"Test 1 - Input: {x1.shape}, Output: {output1.shape}")
    
    # Test 2: Config [512, False, 0.25]
    x2 = torch.randn(2, 256, 32, 32)
    hrfa2 = HRFA(c1=256, c2=512, enhanced_mode=False, reduction_ratio=0.25)
    output2 = hrfa2(x2)
    print(f"Test 2 - Input: {x2.shape}, Output: {output2.shape}")
    
    # Test 3: Config [512, True]
    x3 = torch.randn(2, 512, 16, 16)
    hrfa3 = HRFA(c1=512, c2=512, enhanced_mode=True)
    output3 = hrfa3(x3)
    print(f"Test 3 - Input: {x3.shape}, Output: {output3.shape}")
    
    # Test 4: Config [1024, True]
    x4 = torch.randn(2, 1024, 8, 8)
    hrfa4 = HRFA(c1=1024, c2=1024, enhanced_mode=True)
    output4 = hrfa4(x4)
    print(f"Test 4 - Input: {x4.shape}, Output: {output4.shape}")
    
    print(f"\nParameters (enhanced): {sum(p.numel() for p in hrfa3.parameters()):,}")
    print(f"Parameters (basic): {sum(p.numel() for p in hrfa1.parameters()):,}")