import os
import torch
from pathlib import Path

from cache_io import compare_config

def create_argdict(cfg, model, optimizer, argdict):
    CACHE_ROOT = Path(cfg.cache_root)
    UUID = cfg.uuid
    argdict['log_dir'] = CACHE_ROOT / "pytorch_models" / UUID
    if not argdict['log_dir'].exists():
        argdict['log_dir'].mkdir(parents=True)
        print(f"Created Model Cache Dir:[{argdict['log_dir']}]")
    if not('lr' in argdict):
        argdict['lr'] = optimizer.param_groups[0]['lr']
    argdict['model_state_dict'] = model.state_dict()
    argdict['optimizer_state_dict'] = optimizer.state_dict()
    return argdict

def resume_training(cfg, model, optimizer, argdict):
    """
    Resumes previous training or starts anew

    """
    argdict = create_argdict(cfg, model, optimizer, argdict)
    resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
    if os.path.isfile(resumef):
        checkpoint = torch.load(resumef)
        print("> Resuming previous training")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # -- unpack argdict --
        argdict = checkpoint['args']
        lr = argdict['lr']

        # -- exp config (should be same as "cfg") --
        exp_cfg = checkpoint['exp_cfg']
        start_epoch = exp_cfg.epoch

        # -- get results --
        results = checkpoint['results']

        # -- print training info --
        print("==> checkpoint info")
        keys = ['global_step','epoch']
        for k in keys:
            print("\t{}, {}".format(k,exp_cfg[k]))

    else:
        start_epoch = 0
        exp_cfg = cfg
        results = {}
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0

    # -- update "modifiable" parameters --
    cfg.global_step = exp_cfg.global_step
    cfg.epoch = exp_cfg.epoch
    exp_cfg.device = cfg.device
    exp_cfg.pid = cfg.pid

    # -- assert uuids match --
    match = compare_config(exp_cfg,cfg)
    assert match is True,"configs do not match"

    return start_epoch,results

def save_model_checkpoint(cfg, model, optimizer, results, argdict):
    """
    Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
    Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
    """
    argdict = create_argdict(cfg, model, optimizer, argdict)
    torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'exp_cfg': cfg,
        'args': argdict,
        'results': results
    }
    torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))
    # -- write with epoch count for historic records --
    torch.save(save_dict, os.path.join(argdict['log_dir'],
                                       'ckpt_e{}.pth'.format(cfg.epoch+1)))
    del save_dict

