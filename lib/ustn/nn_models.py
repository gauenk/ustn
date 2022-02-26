
from pathlib import Path
import torch as th
import torch.nn as nn
import torch.optim as optim
from .unet import UNet,UNet_n2n,UNet_v2
from .dncnn import DnCNN
from .red import REDNet20

def get_nn_model(cfg):
    nn_arch = cfg.nn_arch
    if nn_arch == "unet":
        return get_unet_model(cfg)
    elif nn_arch == "dncnn":
        return get_dncnn_model(cfg)
    elif nn_arch == "red":
        return get_dncnn_model(cfg)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def init_model(model,arch,ifile,skips=None):
    if ifile is None: return
    path = Path("./models/") / arch / ifile
    if not(path.exists()): return
    state = th.load(path)
    if not(skips is None):
        state = {k:v for k,v in state.items() if not(k in skips)}
    model.load_state_dict(state,False)

def get_dncnn_model(cfg):
    color = 3 if cfg.color else 1
    model = nn.DataParallel(DnCNN(color,17))
    # skips = ['module.dncnn.0.weight','module.dncnn.47.weight']
    skips = None
    init_model(model,"dncnn",cfg.nn_init,skips)
    model = model.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    init_argdict = {}
    return model,optimizer,init_argdict

def get_unet_model(cfg):
    model = UNet_v2(3)
    # model = UNet_n2n(1)
    init_model(model,"unet_v2",cfg.nn_init)
    model = model.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    init_argdict = {}
    return model,optimizer,init_argdict

