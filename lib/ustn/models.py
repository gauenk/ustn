
import torch
import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from einops import rearrange,repeat


class SingleSTN(nn.Module):

    def __init__(self, shape, device):
        super(SingleSTN, self).__init__()

        # -- stn type --
        self.stype = "grid"

        # -- total variation --
        self.lam_tv = 0.01

        # -- misc --
        self.shape = shape
        self.device = device
        self.scale = 2
        b,c,h,w = shape

        # -- grid --
        if self.stype == "theta":
            self.init_theta(b,device,th.float)
        elif self.stype == "grid":
            self.init_grid(b,h,w,device,th.float)
        else:
            raise ValueError(f"Uknown stype [{self.stype}]")

        # -- pooling --
        self.apool = nn.AvgPool2d(3,stride=1)
        self.mpool = nn.MaxPool2d(7,stride=1)
        self.upool = nn.Upsample(size=(h,w))

    #
    # -- Misc --
    #

    def apply_pool(self,images,ksize,stride):
        nn_pool = nn.AvgPool2d(ksize,stride=1)
        return nn_pool(images)

    #
    # -- Init Parameters --
    #

    def init_theta(self,b,device,dtype):
        self.theta = th.zeros((b,2)).to(self.device).type(dtype)
        self.theta = nn.parameter.Parameter(self.theta)
        self.f_theta = th.zeros((b,2,3)).to(self.device).type(dtype)
        self.f_theta[:,0,0] += self.theta[:,0]
        self.f_theta[:,1,1] += self.theta[:,1]
        self.f_theta = nn.parameter.Parameter(self.f_theta)

    def init_grid(self,b,h,w,device,dtype):
        vectors = [th.arange(0, s, device=device, dtype=dtype)/s for s in [w,h]]
        vectors = [2*(vectors[i]-0.5) for i in range(len(vectors))]
        grids = th.meshgrid(vectors)
        grid  = th.stack(grids) # y, x, z
        grid  = th.unsqueeze(grid, 0)  #add batch
        self.grid = repeat(grid,'1 c w h -> b c h w',b=b)
        self.grid = nn.parameter.Parameter(self.grid)
        return grid

    #
    # -- Forward Pass @ (\theta) or (Grid) --
    #

    def forward(self, x):
        if self.stype == "theta":
            return self.theta_forward(x)
        if self.stype == "grid":
            return self.grid_forward(x)

    def theta_forward(self,x):
        f_theta = self.f_theta
        grid = F.affine_grid(f_theta, x.size())
        x = F.grid_sample(x, grid, mode="bicubic")
        return x

    def grid_forward(self,x):
        ugrid = self.grid
        ugrid = rearrange(ugrid,'t c h w -> t h w c')
        x = F.grid_sample(x, ugrid, mode="bicubic")
        return x

    #
    # -- Compute Loss Function --
    #

    def align_loss(self, image1, image2, clean=None):

        # -- unpack --
        b,c,h,w = image1.shape

        # -- warp loss --
        warp = self.forward(image2)
        loss_rec = self.rec_loss(warp,image1)

        # -- multiscale loss --
        # warp_s = self.apply_pool(warp,3,3)
        # burst_s = self.apply_pool(burst,3,3)
        # offset = 0#2*(30./255.)**2/9.
        # tloss += self.warp_loss(warp_s,burst_s,offset)

        # -- smooth grid --
        loss_tv = self.loss_tv()

        # -- total loss --
        total_loss = loss_rec + self.lam_tv * loss_tv

        return total_loss

    def loss_tv(self):
        if self.stype == "theta":
            return self.loss_tv_theta()
        elif self.stype == "grid":
            return self.loss_tv_grid()

    def loss_tv_theta(self):
        loss_tv = th.mean(th.abs(self.theta))
        return loss_tv

    def loss_tv_grid(self):
        xloss = th.abs(self.grid[:,:,:-1,:] - self.grid[:,:,1:,:]).mean()
        yloss = th.abs(self.grid[:,:,:,:-1] - self.grid[:,:,:,1:]).mean()
        loss_tv  = xloss + yloss
        return loss_tv

    def rec_loss(self,warp,image):
        # -- scale [1] --
        dwarp = th.mean((warp - image)**2)
        return dwarp

    def wmean(self, burst):

        # -- warp loss --
        warp = self.forward(burst)
        # print("warp.shape: ",warp.shape)
        vals = ((burst[[self.ref]] - warp)**2).mean(1,keepdim=True)

        # -- weights --
        weights = th.exp(-vals/2)
        weights /= weights.sum(0,keepdim=True)
        # print(weights.shape)
        # print(weights)

        # -- pool weights --
        # pweights = self.mpool(weights)
        # weights = self.upool(pweights)

        # -- weighted mean --
        wmean = (weights * warp).sum(0)
        print("wmean.shape: ",wmean.shape)
        return wmean

