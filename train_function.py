import torch
from tqdm import tqdm
from losses import *
from templates import customized_templates, compose_text_with_templates
from utils import clip_normalize

def train_fn(disc_B, disc_A, gen_A, gen_B,
             loader, opt_D_B, opt_D_A, opt_G,
             l1, mse, d_scaler, g_scaler,
             loss_mode, clip_model, preprocess, text_A, text_B, use_clip,
             DEVICE, LAMBDA_CYCLE, LAMBDA_IDENTITY, LAMBDA_CLIP):
    B_reals = 0
    B_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (A, B) in enumerate(loop):
        A = A.to(DEVICE)
        B = B.to(DEVICE)

        # Train Discriminators
        with torch.cuda.amp.autocast():
            ## disc_B <adversarial loss>
            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            B_reals += D_B_real.mean().item()
            B_fakes += D_B_fake.mean().item()

            if loss_mode == "original":
                ## l2 ##
                D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))

            elif loss_mode == "fl":
                ## FLL ##
                D_B_real_loss = loss_l2_log(D_B_real - torch.ones_like(D_B_real), (1, 2, 3))
                D_B_fake_loss = loss_l2_log(D_B_fake - torch.zeros_like(D_B_fake), (1, 2, 3))

            D_B_loss = D_B_real_loss + D_B_fake_loss

            ## disc_A <adversarial loss>
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())

            if loss_mode == "original":
                ## l2 ##
                D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))

            elif loss_mode == "fl":
                ## FLL ##
                D_A_real_loss = loss_l2_log(D_A_real - torch.ones_like(D_A_real), (1, 2, 3))
                D_A_fake_loss = loss_l2_log(D_A_fake - torch.zeros_like(D_A_fake), (1, 2, 3))

            D_A_loss = D_A_real_loss + D_A_fake_loss

            # put it together
            D_loss = (D_B_loss + D_A_loss) / 2

        opt_D_B.zero_grad()
        opt_D_A.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_D_B)
        d_scaler.step(opt_D_A)
        d_scaler.update()

        if use_clip:
            with torch.no_grad():
                ## text features
                template_text_A = compose_text_with_templates(text_A, customized_templates)
                template_text_B = compose_text_with_templates(text_B, customized_templates)

                tokens_A = clip.tokenize(template_text_A).to(DEVICE)
                tokens_B = clip.tokenize(template_text_B).to(DEVICE)

                text_features_A = clip_model.encode_text(tokens_A).detach()
                text_features_B = clip_model.encode_text(tokens_B).detach()

                ## for promp-engineering : text_features.mean(dim=0, keepdim=True) is needed
                text_features_A = text_features_A.mean(dim=0, keepdim=True)
                text_features_B = text_features_B.mean(dim=0, keepdim=True)

                text_features_A /= text_features_A.norm(dim=-1, keepdim=True)
                text_features_B /= text_features_B.norm(dim=-1, keepdim=True)

                ## image features (src)
                image_features_original_B = clip_model.encode_image(clip_normalize(B, DEVICE))
                image_features_original_B /= (image_features_original_B.clone().norm(dim=-1, keepdim=True))

                image_features_original_A = clip_model.encode_image(clip_normalize(A, DEVICE))
                image_features_original_A /= (image_features_original_A.clone().norm(dim=-1, keepdim=True))

        # Train Generators night and day
        with torch.cuda.amp.autocast():
            if use_clip:
                ## CLIP directional loss
                ## image directional vector: original B -> generated A
                image_features_generated_A = clip_model.encode_image(clip_normalize(fake_A, DEVICE))
                image_features_generated_A /= (image_features_generated_A.clone().norm(dim=-1, keepdim=True))
                image_dir_B2A = (image_features_generated_A - image_features_original_B)
                image_dir_B2A /= image_dir_B2A.clone().norm(dim=-1, keepdim=True)

                ## image directional vector: original A -> generated B
                image_features_generated_B = clip_model.encode_image(clip_normalize(fake_B, DEVICE))
                image_features_generated_B /= (image_features_generated_B.clone().norm(dim=-1, keepdim=True))
                image_dir_A2B = (image_features_generated_B - image_features_original_A)
                image_dir_A2B /= image_dir_A2B.clone().norm(dim=-1, keepdim=True)

                ## text directional vector: B->A
                text_dir_B2A = (text_features_A - text_features_B)
                text_dir_B2A /= text_dir_B2A.norm(dim=-1, keepdim=True)

                ## text directional vector: A->B
                text_dir_A2B = (text_features_B - text_features_A)
                text_dir_A2B /= text_dir_A2B.norm(dim=-1, keepdim=True)

                ## directional loss
                loss_dir_B2A = (1 - torch.cosine_similarity(image_dir_B2A, text_dir_B2A, dim=1))
                loss_dir_A2B = (1 - torch.cosine_similarity(image_dir_A2B, text_dir_A2B, dim=1))

            # adversarial loss for both generators
            D_B_fake = disc_B(fake_B)
            D_A_fake = disc_A(fake_A)

            if loss_mode == "original":
                ## l2 ##
                loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
                loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

            elif loss_mode == "fl":
                ## FLL ##
                loss_G_B = loss_l2_log(D_B_fake - torch.ones_like(D_B_fake), (1, 2, 3))
                loss_G_A = loss_l2_log(D_A_fake - torch.ones_like(D_A_fake), (1, 2, 3))

            # cycle loss
            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)

            if loss_mode == "original":
                cycle_A_loss = l1(A, cycle_A)
                cycle_B_loss = l1(B, cycle_B)

            elif loss_mode == "fl":
                cycle_A_loss = loss_l1_log(A - cycle_A, (1, 2, 3))
                cycle_B_loss = loss_l1_log(B - cycle_B, (1, 2, 3))

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_A = gen_A(A)
            identity_B = gen_B(B)
            identity_A_loss = l1(A, identity_A)
            identity_B_loss = l1(B, identity_B)

            # add all together
            if use_clip:
                G_loss = (
                        loss_G_A
                        + loss_G_B
                        + cycle_A_loss * LAMBDA_CYCLE
                        + cycle_B_loss * LAMBDA_CYCLE
                        + identity_B_loss * LAMBDA_IDENTITY
                        + identity_A_loss * LAMBDA_IDENTITY
                        + loss_dir_B2A * LAMBDA_CLIP
                        + loss_dir_A2B * LAMBDA_CLIP
                )
            else:
                G_loss = (
                        loss_G_A
                        + loss_G_B
                        + cycle_A_loss * LAMBDA_CYCLE
                        + cycle_B_loss * LAMBDA_CYCLE
                        + identity_B_loss * LAMBDA_IDENTITY
                        + identity_A_loss * LAMBDA_IDENTITY
                )

        opt_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        loop.set_postfix(B_real=B_reals / (idx + 1), B_fake=B_fakes / (idx + 1))