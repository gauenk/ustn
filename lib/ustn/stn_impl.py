
# -- torch --
import torch as th
import torch.optim as optim

# -- local imports --
from .models import SingleSTN

def compute_warped(image1, image2):

    # -- create model --
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
        if i % 100 == 0:
            print("loss.item(): ",loss.item())

        # -- update --
        loss.backward()
        optimizer.step()

        # -- dloss --
        dloss = abs(loss.item() - loss_prev)
        loss_prev = loss.item()
        if dloss < 1e-8: break
    warped = model(image2).detach()
    R = model.parameters()
    return warped,R
