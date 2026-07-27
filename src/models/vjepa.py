import copy
from src.models.encoder import VisionTransformer
from src.models.predictor import Predictor
from src.data.masking import multiblock_mask
import torch.nn as nn
import torch

class VJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = VisionTransformer()
        self.target_encoder = copy.deepcopy(self.context_encoder) # context and target start with identical weights
        
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.predictor = Predictor()
    
    @torch.no_grad()
    def update_target(self, m):
            for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                pt.data.mul_(m).add_(pc.data, alpha=1-m)
                
    def forward(self, x, mask):
        visible_idx = torch.where(~mask)[0]
        masked_idx = torch.where(mask)[0]
        
        # Context encoder - predictor sequence
        ctx = self.context_encoder(x, visible_idx)
        pred = self.predictor(ctx, visible_idx, masked_idx)
        
        # Target encoder
        with torch.no_grad():
            targ = self.target_encoder(x)
            targ = targ[:, masked_idx, :]
        return pred, targ

if __name__ == "__main__":
    model = VJEPA()
    x = torch.randn(2, 3, 16, 112, 112)
    mask = multiblock_mask((8,7,7))
    pred, target = model(x, mask)
    print(pred.shape, target.shape)   # both [2, N_mask, 384], identical shapes
    print("shapes match:", pred.shape == target.shape)