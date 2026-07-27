import torch
from torch import nn
from src.models.attention import TransformerBlock
from src.data.masking import multiblock_mask

class Predictor(nn.Module):
    def __init__(self, num_tokens=392, encoder_dim=384, predictor_dim=192, depth=6, num_heads=6):
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.depth = depth
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(self.encoder_dim, self.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_dim))
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, self.predictor_dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(self.predictor_dim, self.num_heads) for I in range(self.depth)])
        self.norm = nn.LayerNorm(self.predictor_dim)
        self.output_proj = nn.Linear(self.predictor_dim, self.encoder_dim)
    
    def forward(self, ctx, visible_idx, masked_idx):
        # ctx:         [B, N_vis, encoder_dim]      visible token representations from context encoder
        # visible_idx: LongTensor [N_vis]           their positions in the 0..N-1 grid
        # masked_idx:  LongTensor [N_mask]          positions to predict
        # returns:     pred [B, N_mask, encoder_dim]
        ctx_p = self.input_proj(ctx)
        ctx_p = ctx_p + self.pos[:, visible_idx, :]
        N_vis = ctx.shape[1]
        N_masked = self.num_tokens - N_vis
        B = ctx.shape[0]
        m = self.mask_token.expand(B, len(masked_idx), -1)
        m = m + self.pos[:, masked_idx, :]
        seq = torch.cat((ctx_p, m), dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq_out = self.norm(seq)
        seq_out_masked  = seq_out[:, -N_masked:, :]
        out = self.output_proj(seq_out_masked)
        return out
        
if __name__ == "__main__":
    grid = (8, 7, 7)
    mask = multiblock_mask(grid)
    x = torch.randn(2, 392, 384)
    x_vis = x[:, ~mask, :]
    visible_idx = torch.where(~mask)[0]
    masked_idx = torch.where(mask)[0]
    print(x_vis.shape)
    pred = Predictor()
    y = pred(x_vis, visible_idx, masked_idx)
    print(y.shape)
        