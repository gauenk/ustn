
# -- torch --
import torch as th
import torch.optim as optim

# -- local imports --
from .models import SingleSTN

def compute_warped(image1, image2, image2warp=None):

    # -- create model --
    if image2warp is None: image2warp = image2
    model = SingleSTN(image1.shape, image1.device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # -- run steps --
    loss_prev = 10000
    niters = 1000
    for i in range(niters):

        # -- scheduler --
        if (i % 10) == 0 and i > 100:
            scheduler.step()

        # -- zero grad --
        optimizer.zero_grad()
        model.zero_grad()

        # -- state loss --
        loss = model.align_loss(image1,image2)

        # -- dloss --
        dloss = loss_prev - loss.item()
        loss_prev = loss.item()
        # if i % 300 == 0:
        #     print("loss: ",loss.item())
            # print("dloss: ",dloss)

        # -- conditional stops --
        if abs(dloss) < 1e-8: break
        if dloss < -1e-2: break

        # -- update --
        loss.backward()
        optimizer.step()

    warped = model(image2warp).detach()
    params = None
    if model.stype == "grid":
        params = model.grid.data
    return warped,params
