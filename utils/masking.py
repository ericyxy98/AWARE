import torch

def generate_mask(x, r, mean_len=None):
    r"""
    By Xiangyu Yin
    
    Generate a felxible mask that covers a number of segments of the sequence, 
    Overall masking rate is defined by r, and the mean of the masked segment
    lengths is defined by mean_len, The sequence dimension should be placed at 
    last.
    
    This function uses a state-transition method to generate the mask given the
    mean of mask lengths. 
    """
    y = x.view(-1, x.size(-1))
    n = x.size(-1)
    p10 = r/(1-r)/mean_len
    p01 = 1/mean_len
    mask = torch.zeros_like(y)
    rand = torch.rand_like(y)
    for i in range(y.size(0)):
        mask[i,0] = 0 if rand[i,0] < r else 1
        for j in range(1, y.size(1)):
            if mask[i,j-1] == 0:
                mask[i,j] = 1 if rand[i,j] < p01 else 0
            else:
                mask[i,j] = 0 if rand[i,j] < p10 else 1
    mask = mask.view(x.size())

    return mask