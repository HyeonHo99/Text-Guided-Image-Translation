import torch
import torch.optim as optim
import config
from generator import Generator
import numpy as np

def lr_linear_decay_func(epoch):
    if epoch < 100:
        return 2
    else:   # 100<= epoch <200
        decay_amount = config.LEARNING_RATE * ((epoch-99)/100)
        return config.LEARNING_RATE - decay_amount

if __name__ == "__main__":
    G = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    G2 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_G = optim.Adam(
        list(G.parameters()) + list(G2.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    # scheduler_gen = optim.lr_scheduler.LambdaLR(optimizer=opt_G, lr_lambda=lr_linear_decay_func, last_epoch=-1)

    for i in range(200):
        opt_G.step()
        print(f"{i}\t\t{opt_G.param_groups[0]['lr']}")
        if i >= 100:
            lrd = 0.0002/100
            for param_group in opt_G.param_groups:
                param_group['lr'] -= lrd
        # print(scheduler_gen.get_last_lr()[0])
        # scheduler_gen.step()