

# -- imports --
from easydict import EasyDict as edict


def default_config():
    # -- create config --
    cfg = edict()
    cfg.pid = None
    cfg.epoch = -1
    cfg.global_step = -1
    cfg.gpuid = 0
    cfg.random_seed = 123
    # cfg.nn_arch = "unet"
    cfg.nn_arch = "dncnn"
    cfg.nepochs = 30
    cfg.test_interval = 3
    cfg.save_interval = 5
    cfg.train_log_interval = 50
    cfg.test_log_interval = 10
    cfg.batch_size = 1
    cfg.ds_name = "davis"
    # cfg.ds_name = "iphone"
    cfg.noise_level = 30.
    cfg.frame_size = [256,256]
    cfg.scale = 0.5
    cfg.patchsize = 7
    cfg.cache_root = "./cache/"
    cfg.device = "cuda:0"


    return cfg
