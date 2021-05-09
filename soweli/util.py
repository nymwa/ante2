import torch

def generate_square_subsequent_mask(trg_size, src_size=None):
    if src_size is None:
        src_size = trg_size

    mask = (torch.triu(torch.ones(src_size, trg_size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

