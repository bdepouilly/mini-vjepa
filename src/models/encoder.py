import torch
from torch import nn
from patch_embed import TubeletEmbedd
from pos_embed import LearnedPosEmbed
from attention import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(self, num_frames=16, resolution=112, in_channels=3, tubelet_size=2,
                 patch_size=16, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.num_frames = num_frames
        self.resolution = resolution
        self.in_channels = in_channels
        self.tubelet_size = tubelet_size
        assert(self.num_frames % self.tubelet_size == 0)
        self.patch_size = patch_size
        assert(self.resolution % self.patch_size == 0)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        assert(self.embed_dim % self.num_heads == 0)
        self.num_tokens = (self.resolution // self.patch_size) * (self.resolution // self.patch_size) * (self.num_frames // self.tubelet_size)
        
        # Modules
        self.patch_embed = TubeletEmbedd(self.in_channels, self.tubelet_size, self.patch_size,
                                         self.embed_dim)
        self.pos_embed = LearnedPosEmbed(self.num_tokens, self.embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(self.embed_dim, self.num_heads) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, x):
        # x = [batch_size, in_channels, n_frames, resolution, resolution]
        tokens, grid = self.patch_embed(x)
        tokens = self.pos_embed(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)
    
if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'using {device} device')
    
    x = torch.randn(2, 3, 16, 112, 112, device=device)
    m = VisionTransformer().to(device)
    out = m(x)
    print(out.shape, sum(p.numel() for p in m.parameters())/1e6, " M params")