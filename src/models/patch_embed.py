import torch
from torch import nn
import einops

class TubeletEmbedd(nn.Module):
    def __init__(self, in_channels=3, tubelet_size=2, patch_size=16, embed_dim=384):
        super().__init__()
        self.in_channels = in_channels
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.kernel_size = (self.tubelet_size, self.patch_size, self.patch_size)
        self.conv_layer = nn.Conv3d(in_channels=self.in_channels, out_channels=self.embed_dim, 
                                    kernel_size=self.kernel_size, stride=self.kernel_size)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        # returns: tokens [B, N, embed_dim],  and the grid (t, h, w)
        assert (x.shape[2] % self.tubelet_size == 0) & (x.shape[3] % self.patch_size == 0) & (x.shape[4] % self.patch_size == 0)
        x = self.conv_layer(x)
        (t, h, w) = x.shape[2:]
        x = einops.rearrange(x, 'b d t h w -> b (t h w) d')
        return (x, (t, h, w))

if __name__ == "__main__":
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'using {device} device')

    model = TubeletEmbedd().to(device)
    x = torch.randn(2, 3, 16, 112, 112, device=device)
    out = model(x)
    print(out[0].shape)
    print(out[1])