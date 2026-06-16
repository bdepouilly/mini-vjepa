import torch
from torch import nn
import einops

class Attention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (self.embed_dim % self.num_heads == 0)
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.qkv(x)
        x = einops.rearrange(x, "b N (three h h_d) -> three b h N h_d", three=3, h=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = einops.rearrange(out, 'b h N h_d -> b N (h h_d)')
        out = self.proj(out)
        return out
        
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, n_heads=6, mlp_ratio=4):
        super().__init__()
        
        # Constants
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert (self.embed_dim % self.n_heads == 0)
        self.mlp_ratio = mlp_ratio
        
        # Modules
        self.attention = Attention(self.embed_dim, self.n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.mlp_ratio * self.embed_dim),
            nn.GELU(),
            nn.Linear(self.mlp_ratio * self.embed_dim, self.embed_dim)
        )
        self.norm1 = nn.modules.LayerNorm(embed_dim)
        self.norm2 = nn.modules.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        
if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'using {device} device')
    
    b = TransformerBlock().to(device)
    x = torch.randn(2, 392, 384, device=device)
    
    print(b(x).shape)