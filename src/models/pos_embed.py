import torch
from torch import nn

class LearnedPosEmbed(nn.Module):
    def __init__(self, num_tokens = 392, embed_dim = 384):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, self.embed_dim) * 0.02)
        
    def forward(self, tokens):
        return tokens + self.pos
        