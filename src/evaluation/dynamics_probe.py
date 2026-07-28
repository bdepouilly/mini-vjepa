import torch
from torch.nn.functional import smooth_l1_loss
from sklearn.metrics import roc_auc_score

def reverse_time(x, generator=None):
    # x: [B, C, T, H, W] -> reverse the T (time)a axis
    return torch.flip(x, dims=[2])

def shuffle_time(x, generator=None):
    # x: [B, C, T, H, W] -> randomly permutes the T (time) axis
    perm = torch.randperm(x.shape[2], generator=generator)
    return x[:, :, perm, :, :]

@torch.no_grad()
def per_clip_error(model, x, mask):
    # returns: [B] tensor, the mean smooth L1 prediction error per clip
    pred, targ = model(x, mask)
    loss = smooth_l1_loss(pred, targ, reduction="none")
    return torch.mean(loss, dim=[1,2])

@torch.no_grad()
def dynamics_auc(model, mask_grid, corrupt_fn, num_batches=50, batch_size=2, device="cpu", generator=None):
    err_fwd_all = []
    err_cor_all = []
    for b in range(num_batches):
        x = torch.randn(batch_size, 3, 16, 112, 112, device=device)
        mask = multiblock_mask(mask_grid, generator=generator)
        
        err_fwd = per_clip_error(model, x, mask)
        err_fwd_all.append(err_fwd)
        err_cor = per_clip_error(model, corrupt_fn(x), mask)
        err_cor_all.append(err_cor)
    
    # Transform into tensors
    err_fwd_all = torch.cat(err_fwd_all)
    err_cor_all = torch.cat(err_cor_all)
    
    # Prepare sklearn inputs
    scores = torch.cat([err_fwd_all, err_cor_all]).cpu().numpy()
    labels = torch.cat([torch.zeros_like(err_fwd_all), torch.ones_like(err_cor_all)]).cpu().numpy()
    
    return roc_auc_score(labels, scores)

if __name__ == "__main__":
    from src.models.vjepa import VJEPA
    from src.data.masking import multiblock_mask
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(2, 3, 16, 112, 112, device=device)
    model = VJEPA().to(device); model.eval()
    grid = (8, 7, 7)
    mask = multiblock_mask(grid)
    
    rev_x = reverse_time(x)
    print(f"Shape after reversal: {rev_x.shape}")
    sh_x = shuffle_time(x)
    print(f"Shape after shuffle: {sh_x.shape}")

    err = per_clip_error(model, x, mask)
    print(f"Error: {err} / Err. shape: {err.shape}")
    
    auc_shuffle  = dynamics_auc(model, (8,7,7), shuffle_time, device=device)
    auc_reverse  = dynamics_auc(model, (8,7,7), reverse_time, device=device)
    print(f"untrained AUC — shuffle: {auc_shuffle:.3f}   reverse: {auc_reverse:.3f}")
    