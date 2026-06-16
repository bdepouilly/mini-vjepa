import torch
import random
import einops

def multiblock_mask(grid, mask_ratio=0.75, min_block=2, max_block=4, generator=None):
    # grid: (t, h, w)
    # returns: boolean tensor of shape [N] = [t*h*w], True = masked = target
    t, h, w = grid
    spat_mask = torch.zeros(h, w, dtype=torch.bool)
    while (spat_mask.float().mean() < mask_ratio):
        bh = torch.randint(low=min_block, high=max_block+1, size=(1,), generator=generator)
        bw = torch.randint(low=min_block, high=max_block+1, size=(1,), generator=generator)
        top_left_corner_h = random.randint(0, h - bh)
        top_left_corner_w = random.randint(0, w - bw)
        spat_mask[top_left_corner_h:top_left_corner_h+bh,
                  top_left_corner_w:top_left_corner_w+bw] = True
    temp_spat_mask = einops.repeat(spat_mask, 'h w -> (t h w)', t=t)
    return temp_spat_mask

if __name__ == '__main__':
    grid = (8, 7, 7)
    mask = multiblock_mask(grid)
    print(mask.shape)
    print(mask.float().mean())
    single_mask = einops.rearrange(mask, '(t h w) -> t h w', t=8, h=7, w=7)
    single_mask = single_mask[0]
    print(single_mask.int())