
import torch.optim as optim
from .unet import UNet_n2n

def get_nn_model(cfg,nn_arch):
    if nn_arch == "unet":
        return get_unet_model(cfg)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_unet_model(cfg):
    model = UNet_n2n(1)
    model = model.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    init_argdict = {}

    return model,optimizer,init_argdict

