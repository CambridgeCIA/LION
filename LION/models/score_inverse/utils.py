import torch
import random
import numpy as np

def set_global_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_checkpoint_eval(path, model, ema, device):
    """
    Load a checkpoint for evaluation.
    
    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dict into.
        ema (EMA): The EMA object to load the state dict into and apply shadow weights.
        device (torch.device): The device to move the model to after loading.
    
    Returns:
        dict: The checkpoint dictionary containing the loaded state.
    """
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema.apply_shadow()
    model.to(device)
    return ckpt