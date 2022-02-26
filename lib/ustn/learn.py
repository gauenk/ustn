
# -- python imports --
import sys
import numpy as np
from einops import repeat,rearrange

# -- pytorch imports --
import torch as th

# -- project imports --
from easydict import EasyDict as edict
from datasets.wrap_image_data import dict_to_device
from pyutils import print_tensor_stats,save_image
from .stn_impl import compute_warped
from .exp_utils import compute_psnrs

def train_model(cfg,data_loader,denoiser,optim):

    tr_info = []
    data_iter = iter(data_loader)
    nbatches = min(1000,len(data_iter))
    for batch_iter in range(nbatches):

        # -- sample from iterator --
        sample = next(data_iter)

        # -- unpack sample --
        device = f'cuda:{cfg.gpuid}'
        dict_to_device(sample,device)
        dyn_noisy = sample['dyn_noisy'] # dynamics and noise
        noisy = dyn_noisy # alias
        dyn_clean = sample['dyn_clean'] # dynamics and no noise
        clean = dyn_clean # alias
        static_noisy = sample['static_noisy'] # no dynamics and noise
        static_clean = sample['static_clean'] # no dynamics and no noise
        flow_gt = sample['ref_flow']
        image_index = sample['index']

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        nframes = T
        ref_t = nframes//2

        print("batch_iter: ",batch_iter)
        # -- apply denoiser --
        for t in range(nframes-1):

            # -- reset gradient --
            denoiser.zero_grad()
            optim.zero_grad()

            # -- forward --
            image1 = noisy[t]
            image2 = noisy[t+1]
            with th.no_grad():
                deno1 = denoiser(image1).detach()
                deno2 = denoiser(image2).detach()
            # warp2,reg2to1 = compute_warped(clean[t],clean[t+1],clean[t+1])#deno1,deno2)
            # warp2,reg2to1 = compute_warped(clean[t],clean[t+1],noisy[t+1])#deno1,deno2)
            # warp2,reg2to1 = compute_warped(noisy[t],noisy[t+1],noisy[t+1])#deno1,deno2)
            warp2,reg2to1 = compute_warped(deno1,deno2,noisy[t+1])#deno1,deno2)
            # print(reg2to1.shape)
            # warp2 = clean[t]
            deno1 = image1 - denoiser(image1-0.5)
            loss = th.mean((deno1 - warp2)**2)

            # -- backward --
            loss.backward()
            optim.step()

            # -- log --
            if batch_iter % cfg.train_log_interval == 0 and t == 0:
                print("[deno] loss: ",loss.item())
                psnrs = compute_psnrs(deno1,clean[t])
                print("[deno] psnr: ",psnrs)#.mean().item())
                psnrs = compute_psnrs(noisy[t],clean[t])
                print("[noisy] psnr: ",psnrs)#.mean().item())

                # info = get_train_log_info(cfg,model,denoised,loss,dyn_noisy,
                #                           dyn_clean,sims,masks,aligned,
                #                           flow,flow_gt)
                # info['global_iter'] = cfg.global_step
                # info['batch_iter'] = batch_iter
                # info['mode'] = 'train'
                # info['loss'] = loss.item()
                # print_train_log_info(info,nbatches)

                # # -- save example to inspect --
                # denoised = denoised.detach()
                # with th.no_grad():
                #     inputs = th.clip(inputs,0,1)
                #     target = th.clip(target,0,1)
                #     denoised = th.clip(denoised,0,1)
                #     save_image(f"inputs_{batch_iter}.png",inputs)
                #     save_image(f"target_{batch_iter}.png",target)
                #     save_image(f"denoised_{batch_iter}.png",denoised)

                # tr_info.append(info)

            # -- update global step --
            cfg.global_step += 1

            # -- print update --
            sys.stdout.flush()

    return tr_info

def test_model(cfg,test_loader,denoiser):

    denoiser = denoiser.to(cfg.device)
    test_iter = iter(test_loader)
    nbatches = min(100,len(test_iter))
    psnrs = np.zeros( ( nbatches, cfg.batch_size ) )
    use_record = False
    te_info = []

    with th.no_grad():
        for batch_iter in range(nbatches):

            # -- load data --
            device = f'cuda:{cfg.gpuid}'
            sample = next(test_iter)
            dict_to_device(sample,device)

            # -- unpack --
            dyn_noisy = sample['dyn_noisy']
            dyn_clean = sample['dyn_clean']
            noisy,clean = dyn_noisy,dyn_clean
            static_noisy = sample['static_noisy']
            flow_gt = sample['ref_flow']
            nframes = dyn_clean.shape[0]
            T,B,C,H,W = dyn_noisy.shape

            #
            # -- denoise image --
            #

            deno = th.zeros_like(noisy)
            for t in range(nframes):
                deno[t] = noisy[t] - denoiser(noisy[t]-0.5)

            # -- rec loss --
            loss = th.mean((deno - clean)**2)
            psnrs = compute_psnrs(deno,clean)

            # -- log info --
            info = edict()
            info['batch_psnrs'] = psnrs.squeeze()
            info['global_iter'] = cfg.global_step
            info['batch_iter'] = batch_iter
            info['mode'] = 'test'
            info['loss'] = loss.item()
            # te_info.append(info)

            # -- print to screen --
            if batch_iter % cfg.test_log_interval == 0:
                psnr = info['batch_psnrs'].mean().item()
                print("[%d/%d] Test PSNR: %2.2f" % (batch_iter+1,nbatches,psnr))

            # -- print update --
            sys.stdout.flush()

    # -- print final update --
    print("[%d/%d] Test PSNR: %2.2f" % (batch_iter+1,nbatches,psnr))
    sys.stdout.flush()


    return te_info

