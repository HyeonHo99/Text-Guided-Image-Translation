import random, torch, os, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset import CycleGANDataset
from torch.utils.data import DataLoader
import torchvision.utils
from tqdm import tqdm
import copy

def save_checkpoint(model, optimizer, filename="./checkpoints/my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr

def load_checkpoint_model(checkpoint_file,model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])

def seed_everything(seed=2022):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data,0.0,0.02)

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def lr_decay(optim_list, lrd):
    for optim in optim_list:
        for param_group in optim.param_groups:
            param_group['lr'] -= lrd

# def lr_linear_decay_func(epoch):
#     if epoch < 100:
#         return config.LEARNING_RATE
#     else:   # 100<= epoch <200
#         decay_amount = config.LEARNING_RATE * ((epoch-99)/100)
#         return config.LEARNING_RATE - decay_amount

def get_dataset_loader(root_A, root_B, transform, batch_size, shuffle):
    dataset = CycleGANDataset(
        root_A=root_A, root_B=root_B,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True
    )

    return dataset, loader

def record_on_tensorboard(G_A,G_B,train_loader,test_loader,train_writers,test_writers,step,grid_save_dir):
    train_writer_fake_A, train_writer_fake_B = train_writers
    test_writer_fake_A, test_writer_fake_B = test_writers

    if step==1:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for A, B in train_loader:
                    A = A.to("cuda")
                    B = B.to("cuda")

                    img_grid_original_A = torchvision.utils.make_grid(A * 0.5 + 0.5, nrow=4, normalize=False)
                    img_grid_original_B = torchvision.utils.make_grid(B * 0.5 + 0.5, nrow=4, normalize=False)

                    torchvision.utils.save_image(img_grid_original_A, grid_save_dir + "/train/grid/grid_A.png")
                    torchvision.utils.save_image(img_grid_original_B, grid_save_dir + "/train/grid/grid_B.png")
                    break

                for A, B in test_loader:
                    A = A.to("cuda")
                    B = B.to("cuda")

                    img_grid_original_A = torchvision.utils.make_grid(A * 0.5 + 0.5, nrow=4, normalize=False)
                    img_grid_original_B = torchvision.utils.make_grid(B * 0.5 + 0.5, nrow=4, normalize=False)

                    torchvision.utils.save_image(img_grid_original_A, grid_save_dir + "/test/grid/grid_A.png")
                    torchvision.utils.save_image(img_grid_original_B, grid_save_dir + "/test/grid/grid_B.png")
                    break
        print("Saved original input images")

    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for A, B in train_loader:
                    A = A.to("cuda")
                    B = B.to("cuda")

                    fake_B = G_B(A).reshape(-1, 3, 256, 256)
                    fake_A = G_A(B).reshape(-1, 3, 256, 256)

                    img_grid_fake_A = torchvision.utils.make_grid(fake_A * 0.5 + 0.5, nrow=4, normalize=False)
                    img_grid_fake_B = torchvision.utils.make_grid(fake_B * 0.5 + 0.5, nrow=4, normalize=False)

                    train_writer_fake_A.add_image("B2A", img_grid_fake_A, global_step=step)
                    train_writer_fake_B.add_image("A2B", img_grid_fake_B, global_step=step)
                    break

                for A,B in test_loader:
                    A = A.to("cuda")
                    B = B.to("cuda")

                    fake_B = G_B(A).reshape(-1, 3, 256, 256)
                    fake_A = G_A(B).reshape(-1, 3, 256, 256)

                    img_grid_fake_A = torchvision.utils.make_grid(fake_A * 0.5 + 0.5, nrow=4, normalize=False)
                    img_grid_fake_B = torchvision.utils.make_grid(fake_B * 0.5 + 0.5, nrow=4, normalize=False)

                    test_writer_fake_A.add_image("B2A", img_grid_fake_A, global_step=step)
                    test_writer_fake_B.add_image("A2B", img_grid_fake_B, global_step=step)
                    break

        print(f"Saving on tensorboard finished -- current epoch{step}")

def inference(G_A, G_B, train_loader, test_loader, train_save_dir, test_save_dir):
    G_A.eval()
    G_B.eval()

    loop = tqdm(train_loader, leave=True)
    for idx, (A, B) in enumerate(loop):
        A = A.to("cuda")
        B = B.to("cuda")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fake_B = G_B(A)
                fake_A = G_A(B)

                torchvision.utils.save_image(fake_B * 0.5 + 0.5, train_save_dir + f"/reconstructedB/{idx}.png")
                torchvision.utils.save_image(fake_A * 0.5 + 0.5, train_save_dir + f"/reconstructedA/{idx}.png")

    print(f"Inference of train images finished >> saved on {train_save_dir}")

    loop = tqdm(test_loader, leave=True)
    for idx, (A, B) in enumerate(loop):
        A = A.to("cuda")
        B = B.to("cuda")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fake_B = G_B(A)
                fake_A = G_A(B)

                torchvision.utils.save_image(fake_B * 0.5 + 0.5, test_save_dir + f"/reconstructedB/{idx}.png")
                torchvision.utils.save_image(fake_A * 0.5 + 0.5, test_save_dir + f"/reconstructedA/{idx}.png")

    print(f"Inference of test images finished >> saved on {test_save_dir}")
    G_A.train()
    G_B.train()


