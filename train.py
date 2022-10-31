import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from cleanfid import fid
import torchvision.utils
import clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *
from prompt_edit import edit
from discriminator import Discriminator
from generator import Generator
from train_function import train_fn
from dataset import CycleGANDataset_single


'''
    python train.py --mode fl-clip --data horse2zebra --epoch 200 --fid True --gpu 0
                    --batch_size 1 --lr 0.0002 --lr_decay True 
                    --lambda_identity 0.1 --lambda_cycle 10 --lambda_clip 1 --num_workers 1
                    --tensorboard_per 10 --checkpoint_per 50
                    --load_model False --save_model True
                    --img_size 256
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--fid', type=bool, default=False)
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--lambda_identity', type=float, default=0.1)
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--lambda_clip', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--tensorboard_per', type=int, default=10)
    parser.add_argument('--checkpoint_per', type=int, default=50)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=256)


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Set the GPU 0 to use

    dataset_dir = os.path.join("dataset", args.data)
    INFER_DIR = os.path.join("results", args.data, args.mode)
    EVAL_DIR = os.path.join("evaluations", args.data, args.mode)

    CHECKPOINT_DIR = os.path.join("checkpoints", args.data, args.mode)
    CHECKPOINT_G_B = os.path.join(CHECKPOINT_DIR, "G_B.pth")
    CHECKPOINT_G_A = os.path.join(CHECKPOINT_DIR, "G_A.pth")
    CHECKPOINT_D_B = os.path.join(CHECKPOINT_DIR, "D_B.pth")
    CHECKPOINT_D_A = os.path.join(CHECKPOINT_DIR, "D_A.pth")

    TEXT_A = args.data.split("2")[0]
    TEXT_B = args.data.split("2")[1]

    TEXT_A, TEXT_B = edit([TEXT_A], [TEXT_B])
    TEXT_A = TEXT_A[0]
    TEXT_B = TEXT_B[0]

    DOMAIN_A = args.data.split("2")[0]
    DOMAIN_B = args.data.split("2")[1]

    ## load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)

    print()
    print(f"{args.mode}\t{DOMAIN_A}<->{DOMAIN_B}")

    ## transform function for training
    transforms = A.Compose(
        [
            A.Resize(width=args.img_size, height=args.img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    ## transform function for validation : no HorizontalFlip
    val_transforms = A.Compose(
        [
            A.Resize(width=args.img_size, height=args.img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    eval_transforms = A.Compose(
        [
            A.Resize(width=args.img_size, height=args.img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    ## Dataset, DataLoader
    dataset, loader = get_dataset_loader(root_A=dataset_dir + "/trainA",
                                         root_B=dataset_dir + "/trainB",
                                         transform=transforms, batch_size=args.batch_size, shuffle=True)

    val_seen_dataset, val_seen_loader = get_dataset_loader(root_A=dataset_dir + "/trainA",
                                                           root_B=dataset_dir + "/trainB",
                                                           transform=val_transforms, batch_size=16,
                                                           shuffle=False)

    val_unseen_dataset, val_unseen_loader = get_dataset_loader(root_A=dataset_dir + "/testA",
                                                               root_B=dataset_dir + "/testB",
                                                               transform=val_transforms, batch_size=16,
                                                               shuffle=False)


    ## datasets and dataloaders for FID calculation
    ## A2B [train]
    A2B_train_dataset = CycleGANDataset_single(root=dataset_dir + "/trainA",
                                               transform=eval_transforms)
    A2B_train_loader = DataLoader(
        A2B_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    ## A2B [test]
    A2B_test_dataset = CycleGANDataset_single(root=dataset_dir + "/testA",
                                              transform=eval_transforms)
    A2B_test_loader = DataLoader(
        A2B_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    ## B2A [train]
    B2A_train_dataset = CycleGANDataset_single(root=dataset_dir + "/trainB",
                                               transform=eval_transforms)
    B2A_train_loader = DataLoader(
        B2A_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    ## B2A [test]
    B2A_test_dataset = CycleGANDataset_single(root=dataset_dir + "/testB",
                                              transform=eval_transforms)
    B2A_test_loader = DataLoader(
        B2A_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    ## Gradient Scaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    ###################################
    ######### MAIN TRAINING ###########
    ###################################
    D_B = Discriminator(in_channels=3).to(DEVICE)
    D_A = Discriminator(in_channels=3).to(DEVICE)
    G_A = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    G_B = Generator(img_channels=3, num_residuals=9).to(DEVICE)

    ## initialize weights from ~N(0,0.02)
    initialize_weights(D_B)
    initialize_weights(D_A)
    initialize_weights(G_A)
    initialize_weights(G_B)

    opt_D_A = optim.Adam(
        list(D_A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_D_B = optim.Adam(
        list(D_B.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_G = optim.Adam(
        list(G_A.parameters()) + list(G_B.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    if args.load_model:
        load_checkpoint(
            CHECKPOINT_G_B, G_B, opt_G)
        load_checkpoint(
            CHECKPOINT_G_A, G_A, opt_G)
        load_checkpoint(
            CHECKPOINT_D_B, D_B, opt_D_B)
        load_checkpoint(
            CHECKPOINT_D_A, D_A, opt_D_A)

    ## checkpoint writers
    test_writer_fake_A = SummaryWriter("runs/test/" + args.data + f"/{args.mode}/B2A")
    test_writer_fake_B = SummaryWriter("runs/test/" + args.data + f"/{args.mode}/A2B")
    train_writer_fake_A = SummaryWriter("runs/train/" + args.data + f"/{args.mode}/B2A")
    train_writer_fake_B = SummaryWriter("runs/train/" + args.data + f"/{args.mode}/A2B")

    step = 1
    ## save input grid image
    record_on_tensorboard(G_A, G_B, val_seen_loader, val_unseen_loader,
                          [train_writer_fake_A, train_writer_fake_B],
                          [test_writer_fake_A, test_writer_fake_B],
                          step=1, grid_save_dir=INFER_DIR)

    ## mode setting
    if args.mode == "original":
        loss_mode = "original"
        use_clip = False
    elif args.mode == "fl":
        loss_mode = "fl"
        use_clip = False
    elif args.mode == "clip":
        loss_mode = "original"
        use_clip = True
    elif args.mode == "fl-clip":
        loss_mode = "fl"
        use_clip = True
    else:
        print(f"<ERROR>\tThere is no such mode:\t{args.mode}")

    ## main training
    for epoch in range(args.epoch):
        train_fn(D_B, D_A, G_A, G_B, loader, opt_D_B, opt_D_A,
                 opt_G, nn.L1Loss, nn.MSELoss, d_scaler, g_scaler, loss_mode,
                 clip_model, preprocess, TEXT_A, TEXT_B, use_clip=use_clip,
                 DEVICE=DEVICE, LAMBDA_CYCLE=args.lambda_cycle, LAMBDA_IDENTITY=args.lambda_identity,
                 LAMBDA_CLIP=args.lambda_clip)

        ## linear decay [100,200] : 0.0002 -> 0
        if args.lr_decay and epoch >= 100:
            lr_decay([opt_D_A, opt_D_B, opt_G], lrd=args.lr / 100)

        ## save on tensorboard
        if (epoch + 1) % args.tensorboard_per == 0:
            record_on_tensorboard(G_A, G_B, val_seen_loader, val_unseen_loader,
                                  [train_writer_fake_A, train_writer_fake_B],
                                  [test_writer_fake_A, test_writer_fake_B],
                                  step=step, grid_save_dir=None)
        step += 1

        ## save checkpoints
        if args.save_model and (epoch + 1) % args.checkpoint_per == 0:
            save_checkpoint(G_B, opt_G, filename=CHECKPOINT_G_B)
            save_checkpoint(G_A, opt_G, filename=CHECKPOINT_G_A)
            save_checkpoint(D_B, opt_D_B, filename=CHECKPOINT_D_B)
            save_checkpoint(D_A, opt_D_A, filename=CHECKPOINT_D_A)


        if args.fid:
            ## calculate FID per every epoch
            G_A.eval()
            G_B.eval()

            ## A2B [train]
            save_dir = EVAL_DIR + "/train"
            with torch.no_grad():
                for idx, A in enumerate(A2B_train_loader):
                    A = A.to(DEVICE)
                    reconstructed_B = G_B(A)
                    torchvision.utils.save_image(reconstructed_B * 0.5 + 0.5, save_dir + f"/reconstructedB/{idx}.png")

            ## A2B [test]
            save_dir = EVAL_DIR + "/test"
            with torch.no_grad():
                for idx, A in enumerate(A2B_test_loader):
                    A = A.to(DEVICE)
                    reconstructed_B = G_B(A)
                    torchvision.utils.save_image(reconstructed_B * 0.5 + 0.5, save_dir + f"/reconstructedB/{idx}.png")

            ## B2A [train]
            save_dir = EVAL_DIR + "/train"
            with torch.no_grad():
                for idx, B in enumerate(B2A_train_loader):
                    B = B.to(DEVICE)
                    reconstructed_A = G_A(B)
                    torchvision.utils.save_image(reconstructed_A * 0.5 + 0.5, save_dir + f"/reconstructedA/{idx}.png")

            ## B2A [test]
            save_dir = EVAL_DIR + "/test"
            with torch.no_grad():
                for idx, B in enumerate(B2A_test_loader):
                    B = B.to(DEVICE)
                    reconstructed_A = G_A(B)
                    torchvision.utils.save_image(reconstructed_A * 0.5 + 0.5, save_dir + f"/reconstructedA/{idx}.png")

            ## eval: FID - pytorch / FID - clean / KID
            fid_original_A2B_train = fid.compute_fid(fdir1=dataset_dir + "/trainB",
                                                     fdir2=EVAL_DIR + "/train/reconstructedB",
                                                     mode="legacy_pytorch")
            fid_original_A2B_test = fid.compute_fid(fdir1=dataset_dir + "/testB",
                                                    fdir2=EVAL_DIR + "/test/reconstructedB",
                                                    mode="legacy_pytorch")
            fid_original_B2A_train = fid.compute_fid(fdir1=dataset_dir + "/trainA",
                                                     fdir2=EVAL_DIR + "/train/reconstructedA",
                                                     mode="legacy_pytorch")
            fid_original_B2A_test = fid.compute_fid(fdir1=dataset_dir + "/testA",
                                                    fdir2=EVAL_DIR + "/test/reconstructedA",
                                                    mode="legacy_pytorch")

            ## record on .txt
            with open(EVAL_DIR + f"/{DOMAIN_A}2{DOMAIN_B}_train.txt", "a") as f:
                f.write(f"{epoch + 1}\t{fid_original_A2B_train}\n")
            with open(EVAL_DIR + f"/{DOMAIN_A}2{DOMAIN_B}_test.txt", "a") as f:
                f.write(f"{epoch + 1}\t{fid_original_A2B_test}\n")
            with open(EVAL_DIR + f"/{DOMAIN_B}2{DOMAIN_A}_train.txt", "a") as f:
                f.write(f"{epoch + 1}\t{fid_original_B2A_train}\n")
            with open(EVAL_DIR + f"/{DOMAIN_B}2{DOMAIN_A}_test.txt", "a") as f:
                f.write(f"{epoch + 1}\t{fid_original_B2A_test}\n")

            ## save fid on tensorboard
            train_writer_fake_B.add_scalar("FID", fid_original_A2B_train, global_step=epoch + 1)
            test_writer_fake_B.add_scalar("FID", fid_original_A2B_test, global_step=epoch + 1)
            train_writer_fake_A.add_scalar("FID", fid_original_B2A_train, global_step=epoch + 1)
            test_writer_fake_A.add_scalar("FID", fid_original_B2A_test, global_step=epoch + 1)

            G_A.train()
            G_B.train()