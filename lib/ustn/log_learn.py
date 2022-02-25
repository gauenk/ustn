
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat

from pyutils import images_to_psnrs
from align import compute_epe, compute_pair_flow_acc
from align.xforms import align_from_flow

import torch
import torchvision

center_crop = torchvision.transforms.functional.center_crop

def print_train_log_info(info,nbatches):
    msg = f"Train @ [{info['batch_iter']}/{nbatches}]"

    image_psnrs = info['image_psnrs']
    mean = image_psnrs.mean()
    msg += f" [PSNR(i)]: %2.2f" % (mean)

    sim_psnrs = info['sim_psnrs']
    mean = sim_psnrs.mean()
    msg += f" [PSNR(s)]: %2.2f" % (mean)

    aligned_psnrs = info['aligned_psnrs']
    mean = aligned_psnrs.mean()
    msg += f" [PSNR(a)]: %2.2f" % (mean)

    epes = info['epe']
    mean = epes.mean()
    msg += f" [EPE(i)]: %2.1f" % (mean)

    print(msg)


def get_train_log_info(cfg,model,denoised,loss,dyn_noisy,dyn_clean,
                       sims,masks,aligned,flow,flow_gt):

    # -- init info --
    info = {}
    nframes,nimages,ncolor,h,w = dyn_clean.shape
    ref_t = nframes // 2

    # -- image psnrs --
    image_psnrs = images_to_psnrs(denoised,dyn_clean[ref_t])
    info['image_psnrs'] = image_psnrs

    # -- sim images psnrs
    nimages = sims.shape[1]
    nsims = sims.shape[0]-1
    clean = repeat(dyn_clean[ref_t],'b c h w -> s b c h w',s=nsims)
    ref_clean = rearrange(clean,'s b c h w -> (s b) c h w')
    sims = rearrange(sims[1:],'s b c h w -> (s b) c h w')
    sim_psnrs = images_to_psnrs(ref_clean,sims)
    sim_psnrs = rearrange(sim_psnrs,'(t b) -> t b',b=nimages)
    info['sim_psnrs'] = sim_psnrs

    # -- aligned image psnrs --
    pad = cfg.patchsize//2 + cfg.nblocks//2
    T,B,C,H,W = dyn_noisy.shape
    isize = edict({'h':H,'w':W})
    csize = [H-2*pad,W-2*pad]
    clean = repeat(dyn_clean[ref_t],'b c h w -> t b c h w',t=nframes)
    ref_clean = rearrange(clean,'t b c h w -> (t b) c h w')
    if not(flow is None):
        aligned_clean = align_from_flow(dyn_clean,flow,cfg.nblocks,isize=isize)
        aligned_clean = aligned_clean.to(dyn_clean.device,non_blocking=True)
        aligned_rs = rearrange(aligned_clean,'t b c h w -> (t b) c h w')
        ref_clean = center_crop(ref_clean,csize)
        aligned_rs = center_crop(aligned_rs,csize)
        aligned_psnrs = images_to_psnrs(ref_clean,aligned_rs)
        aligned_psnrs = rearrange(aligned_psnrs,'(t b) -> t b',b=B)
        aligned_psnrs = remove_center_frame(aligned_psnrs)
        info['aligned_psnrs'] = aligned_psnrs
    else: info['aligned_psnrs'] = np.zeros(1)

    # -- epe errors --
    if not(flow is None):
        info['epe'] = compute_epe(flow,flow_gt)
    else: info['epe'] = np.zeros(1)

    # -- nnf acc --
    if not(flow is None):
        info['nnf_acc'] = compute_pair_flow_acc(flow,flow_gt)
    else: info['nnf_acc'] = np.zeros(1)

    return info

def remove_center_frame(burst):
    nframes = burst.shape[0]
    ref = nframes//2
    left,right = burst[:ref],burst[ref+1:]
    if torch.is_tensor(burst):
        burst = torch.cat([left,right],dim=0)
    else:
        burst = np.concatenate([left,right],axis=0)
    return burst

def get_test_log_info(cfg,model,denoised,loss,dyn_noisy,dyn_clean):

    # -- init info --
    info = {}
    nframes,nimages,ncolor,h,w = dyn_clean.shape
    ref_t = nframes // 2

    # -- image psnrs --
    image_psnrs = images_to_psnrs(denoised,dyn_clean[ref_t])
    info['image_psnrs'] = image_psnrs

    # -- empty for square matrix later --
    info['aligned_psnrs'] = []
    info['sim_psnrs'] = []
    info['epe'] = []
    info['nnf_acc'] = []

    return info
