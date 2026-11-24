import numpy as np
import matplotlib.pyplot as plt

import torch


def random_rectangle(min_cut=50,image_size=512):
    rangex = np.random.randint(min_cut,image_size-min_cut)
    rangey = np.random.randint(min_cut,image_size-min_cut)

    x_coord = np.arange(-(rangex-1),image_size,1,dtype=int)
    y_coord = np.arange(-(rangey-1),image_size,1, dtype=int)

    x_start = np.random.choice(x_coord)
    y_start = np.random.choice(y_coord)



    mask = np.zeros((image_size,image_size),dtype=int)

    for i in range(y_start,y_start+rangey):
        for j in range(x_start,x_start+rangex):
            if i in range(image_size):
                if j in range(image_size):
                    mask[i][j] = 1



    
    return torch.from_numpy(mask)

def batch_of_masks(mask,dim = 1, h=512,w=512):
    batch_mask = torch.empty((dim, 1, h, w), dtype=torch.float32)

    for i in range(dim):
        batch_mask[i][0] = mask

    return batch_mask


def cut_mix(real, fake, mask, device):
    torch.cuda.set_device(device)
    # print(f"mask {mask.size()}")
    # print(f"real {real.size()}")
    # print(f"fake {fake.size()}")
    return torch.where(mask.bool().to(device), real, fake).to(device)