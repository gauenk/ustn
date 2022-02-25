
import torch.optim as optim
from .unet import UNet,UNet_n2n,UNet_v2
from .dncnn import DnCNN

def get_nn_model(cfg,nn_arch):
    if nn_arch == "unet":
        return get_unet_model(cfg)
    if nn_arch == "dncnn":
        return get_dncnn_model(cfg)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_dncnn_model(cfg):
    model = DnCNN(3)
    model = model.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    init_argdict = {}
    return model,optimizer,init_argdict

def get_unet_model(cfg):
    model = UNet_v2(3)
    # model = UNet_n2n(1)
    model = model.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    init_argdict = {}
    return model,optimizer,init_argdict

