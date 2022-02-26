

# -- python imports --
import time,sys,os
import numpy as np
import pandas as pd
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- multiprocessing --
from multiprocessing import current_process

# -- pytorch imports --
import torch as th

# -- project imports --
import settings
import cache_io
from datasets import load_dataset

# -- cuda profiler --
import nvtx

# -- [local] package imports --
from .train_common import save_model_checkpoint,resume_training
from .exp_utils import *
from .learn import train_model,test_model
from .nn_models import get_nn_model


def denoise_stn(cfg):

    # -- reset sys.out if subprocess --
    cproc = current_process()
    if not(cfg.pid == cproc.pid):
        printfn = Path("./running")  / f"{os.getpid()}.txt"
        orig_stdout = sys.stdout
        f = open(printfn, 'w')
        sys.stdout = f

    # -- init exp! --
    print("RUNNING Exp: [UNSUP DENOISING] Compare to Competitors")
    print("we use the noisy images to warp a noisy image here.")
    print(cfg)
    cfg.epoch = -1
    cfg.global_step = -1

    # -- set default device --
    th.cuda.set_device(cfg.gpuid)

    # -- create results record to save --
    dims={'batch_results':None,'batch_to_record':None,
          'record_results':{'default':0},
          'stack':{'default':0},'cat':{'default':0}}
    record = cache_io.ExpRecord(dims)

    # -- set random seed --
    set_seed(cfg.random_seed)

    # -- get neural netowrk --
    model,optim,argdict = get_nn_model(cfg)

    # -- load dataset --
    data,loaders = load_dataset(cfg)

    # -- check if exists --
    start_epoch,results = resume_training(cfg, model, optim, argdict)
    print(f"Starting from epoch [{start_epoch}]")

    #
    # -- Primary Learning Loop --
    #
    result_te = test_model(cfg,loaders.te,model)
    print(result_te)

    # -- start test --
    start_time = time.perf_counter()
    for epoch in range(start_epoch,cfg.nepochs):
        print("-"*25)
        print(f"Epoch [{epoch}]")
        print("-"*25)
        cfg.epoch = epoch
        result_tr = train_model(cfg,loaders.tr,model,optim)
        append_result_to_dict(results,result_tr)
        if epoch % cfg.test_interval == 0:
            result_te = test_model(cfg,loaders.te,model)
            print("Testing: ",result_te)
            append_result_to_dict(results,result_te)
        if epoch % cfg.save_interval == 0:
            save_model_checkpoint(cfg, model, optim, argdict, results)
    result_te = test_model(cfg,loaders.te,model)
    save_model_checkpoint(cfg, model, optim, argdict)
    append_result_to_dict(results,result_te)
    runtime = time.perf_counter() - start_time

    # -- format results --
    # listdict_to_numpy(results)
    results['runtime'] = np.array([runtime])

    return results
