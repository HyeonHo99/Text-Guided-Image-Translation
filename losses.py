import torch

def loss_l1(x, reduce=None):

    eps = 1e-1
    x_abs = torch.abs(x)
    scale = torch.max(torch.max(x_abs), torch.tensor(1))
    distance = torch.mean(\
        x_abs, dim=reduce)

    return distance

def loss_l1_log(x, reduce=None):

    eps = 1e-1
    x_abs = torch.abs(x)
    scale = torch.max(torch.max(x_abs), torch.tensor(1))
    distance = torch.mean(\
        -torch.log(1-x_abs/scale*(1-eps)), dim=reduce)

    return distance

def loss_l2(x,reduce=None):
    eps = 1e-1
    x_square = torch.square(x)
    scale = torch.max(torch.max(x_square),torch.tensor(1))
    distance = torch.mean(x_square,dim=reduce)

    return distance


def loss_l2_log(x,reduce=None):
    eps = 1e-1
    x_square = torch.square(x)
    scale = torch.max(torch.max(x_square),torch.tensor(1))
    distance = torch.mean(-torch.log(1-x_square/scale*(1-eps)),dim=reduce)

    return distance
