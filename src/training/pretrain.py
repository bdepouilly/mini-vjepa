import torch
import torch.nn.functional as F
from src.models.vjepa import VJEPA
from src.data.masking import multiblock_mask

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VJEPA().to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.AdamW(params, lr=1.5e-4, weight_decay=0.04)
    
    x = torch.randn(2, 3, 16, 112, 112, device=device)
    grid = (8, 7, 7)
    steps = 200
    g = torch.Generator().manual_seed(42)
    
    for step in range(steps):
        mask = multiblock_mask(grid, generator=g)
        pred, targ = model(x, mask)
        loss = F.smooth_l1_loss(pred, targ)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        m = 0.996 + (1.0 - 0.996) * (step / steps)
        model.update_target(m)
        
        if step % 20 == 0:
            print(f"step {step:4d} loss {loss.item():.4f}")
            
if __name__ == "__main__":
    main()
        