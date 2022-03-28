import argparse
from ctypes import resize
import math
import random
import os
import numpy as np
import lpips
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True, target_layer=None):
    for name, param  in model.named_parameters():
        if target_layer is None or target_layer in name:
            param.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def perceptual_loss(loss_fn_vgg, real_img, fake_img):
    p_loss = 0

    # p loss face
    x, y =  50,102
    width, height = 146, 102
    p_loss = p_loss + loss_fn_vgg(real_img[:, :, y:y+height, x:x+width], fake_img[:, :, y:y+height, x:x+width])

    # p loss eyes
    x, y =  50,102
    width, height = 146, 46
    p_loss = p_loss + loss_fn_vgg(real_img[:, :, y:y+height, x:x+width], fake_img[:, :, y:y+height, x:x+width])

    # p loss nose
    x, y =  86,134
    width, height = 86, 46
    p_loss = p_loss + loss_fn_vgg(real_img[:, :, y:y+height, x:x+width], fake_img[:, :, y:y+height, x:x+width])

    # p loss mouth
    x, y =  50,174
    width, height = 146, 46
    p_loss = p_loss + loss_fn_vgg(real_img[:, :, y:y+height, x:x+width], fake_img[:, :, y:y+height, x:x+width])

    return p_loss.mean()

def get_feature(img, feature_type):
    output = img

    if feature_type == 'face':
        x, y =  50,102
        width, height = 146, 102
        output = img[:, :, y:y+height, x:x+width]
    elif feature_type == 'eyes':
        x, y =  50,102
        width, height = 146, 46
        output = img[:, :, y:y+height, x:x+width]
    elif feature_type == 'nose':
        x, y =  86,134
        width, height = 86, 46
        output = img[:, :, y:y+height, x:x+width]
    elif feature_type == 'mouth':
        x, y =  50,174
        width, height = 146, 46
        output = img[:, :, y:y+height, x:x+width]
    
    transform = Resize(size=(256,256))
    return transform(output)

def train(args, loader, generator, generator_source, discriminator, g_optim, d_optim, g_ema, device, feature_discriminators):
    # create directories
    save_dir = args.expr_dir
    os.makedirs(save_dir, 0o777, exist_ok=True)
    os.makedirs(save_dir + "/checkpoints", 0o777, exist_ok=True)

    # create lpips model
    loss_fn_vgg = lpips.PerceptualLoss(net='vgg')

    # create tensorboard log file in experiment directory
    writer = SummaryWriter(save_dir)
    loader = sample_data(loader)
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # definitions
    mean_path_length = 0 # shortest path between a pair of nodes
    
    face_loss_val = 0
    eyes_loss_val = 0
    nose_loss_val = 0
    mouth_loss_val = 0
    d_loss_val = 0 # discriminator loss
    g_loss_val = 0 # generator loss
    r1_loss = torch.tensor(0.0, device=device) # r1 regularization
    path_loss = torch.tensor(0.0, device=device) # path loss
    path_lengths = torch.tensor(0.0, device=device) # path length regularization
    mean_path_length_avg = 0 # mean average path length regularization
    loss_dict = {} # loss dictionary

    # distributed settings settings
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        d_face_module = feature_discriminators['face'].module
        d_eyes_module = feature_discriminators['eyes'].module
        d_nose_module = feature_discriminators['nose'].module
        d_mouth_module = feature_discriminators['mouth'].module
    else:
        g_module = generator
        d_module = discriminator
        d_face_module = feature_discriminators['face']
        d_eyes_module = feature_discriminators['eyes']
        d_nose_module = feature_discriminators['nose']
        d_mouth_module = feature_discriminators['mouth']

    accum = 0.5 ** (32 / (10 * 1000)) # gradient accumulation
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0 # adaptive augmentation
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    # for sampling an image
    # sample_z = torch.randn(args.n_sample, args.latent, device=device)

    # main training loop
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        # load ground truth
        real_img = next(loader)
        real_img = real_img.to(device)

                            ###########################
                            ### Train Discriminator ###
                            ###########################

        requires_grad(generator, False)
        requires_grad(discriminator, False)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(feature_discriminators['face'], True)
        requires_grad(feature_discriminators['eyes'], True)
        requires_grad(feature_discriminators['nose'], True)
        requires_grad(feature_discriminators['mouth'], True)

        # generate noise and fake image with it
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        ## Perform data augmentation if augment is set in arguments
        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img

        # discriminator makes prediction
        fake_pred = discriminator(fake_img) # predict fake image generated from noise
        real_pred = discriminator(real_img_aug) # predict ground truth

        # feature based discrimination
        face_fake_pred = feature_discriminators['face'](get_feature(fake_img, 'face'))
        face_real_pred = feature_discriminators['face'](get_feature(real_img, 'face'))
        eyes_fake_pred = feature_discriminators['eyes'](get_feature(fake_img, 'eyes'))
        eyes_real_pred = feature_discriminators['eyes'](get_feature(real_img, 'eyes'))
        nose_fake_pred = feature_discriminators['nose'](get_feature(fake_img, 'nose'))
        nose_real_pred = feature_discriminators['nose'](get_feature(real_img, 'nose'))
        mouth_fake_pred = feature_discriminators['mouth'](get_feature(fake_img, 'mouth'))
        mouth_real_pred = feature_discriminators['mouth'](get_feature(real_img, 'mouth'))

        face_loss = d_logistic_loss(face_real_pred, face_fake_pred)
        eyes_loss = d_logistic_loss(eyes_real_pred, eyes_fake_pred)
        nose_loss = d_logistic_loss(nose_real_pred, nose_fake_pred)
        mouth_loss = d_logistic_loss(mouth_real_pred, mouth_fake_pred)

        d_loss = d_logistic_loss(real_pred, fake_pred) # sum of average of softplus -real_pred and fake_pred
        d_loss = d_loss + face_loss + eyes_loss + nose_loss + mouth_loss

        # update discriminator weights
        loss_dict["d_face"] = face_loss
        loss_dict["d_eyes"] = eyes_loss
        loss_dict["d_nose"] = nose_loss
        loss_dict["d_mouth"] = mouth_loss
        loss_dict["d"] = d_loss
        loss_dict["face_real_score"] = real_pred.mean()
        loss_dict["face_fake_score"] = fake_pred.mean()
        loss_dict["eyes_real_score"] = real_pred.mean()
        loss_dict["eyes_fake_score"] = fake_pred.mean()
        loss_dict["nose_real_score"] = real_pred.mean()
        loss_dict["nose_fake_score"] = fake_pred.mean()
        loss_dict["mouth_real_score"] = real_pred.mean()
        loss_dict["mouth_fake_score"] = fake_pred.mean()
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        feature_discriminators['face'].zero_grad()
        feature_discriminators['eyes'].zero_grad()
        feature_discriminators['nose'].zero_grad()
        feature_discriminators['mouth'].zero_grad()
        discriminator.zero_grad() # reset gradient
        face_loss.backward(retain_graph=True)
        eyes_loss.backward(retain_graph=True)
        nose_loss.backward(retain_graph=True)
        mouth_loss.backward(retain_graph=True)
        d_loss.backward() # backpropagate
        d_optim.step() # perform single optimization step

        # augmentation to real prediction
        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        # Discriminator regularisation
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

                            ##################################
                            ### End of Train Discriminator ###
                            ##################################

                                #######################
                                ### Train Generator ###
                                #######################
                            
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(feature_discriminators['face'], False)
        requires_grad(feature_discriminators['eyes'], False)
        requires_grad(feature_discriminators['nose'], False)
        requires_grad(feature_discriminators['mouth'], False)

        #--------------------------
        
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        face_fake_pred = feature_discriminators['face'](get_feature(fake_img, 'face'))
        eyes_fake_pred = feature_discriminators['eyes'](get_feature(fake_img, 'eyes'))
        nose_fake_pred = feature_discriminators['nose'](get_feature(fake_img, 'nose'))
        mouth_fake_pred = feature_discriminators['mouth'](get_feature(fake_img, 'mouth'))
        g_loss = g_nonsaturating_loss(fake_pred) + g_nonsaturating_loss(face_fake_pred) + g_nonsaturating_loss(eyes_fake_pred) + g_nonsaturating_loss(nose_fake_pred) + g_nonsaturating_loss(mouth_fake_pred)

        ## perform perceptual loss (disabled temporarily)
        # p_loss = perceptual_loss(loss_fn_vgg, real_img, fake_img)
        # g_loss = g_loss

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        # generator regularization
        if args.freezeG < 0 and g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            g_optim.step()
            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
                                ##############################
                                ### End of Train Generator ###
                                ##############################

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        d_face_loss_val = loss_dict["d_face"].mean().item()
        d_eyes_loss_val = loss_dict["d_eyes"].mean().item()
        d_nose_loss_val = loss_dict["d_nose"].mean().item()
        d_mouth_loss_val = loss_dict["d_mouth"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        face_real_score_val = loss_dict["face_real_score"].mean().item()
        face_fake_score_val = loss_dict["face_fake_score"].mean().item()
        eyes_real_score_vak = loss_dict["eyes_real_score"].mean().item()
        eyes_fake_score_val = loss_dict["eyes_fake_score"].mean().item()
        nose_real_score_val = loss_dict["nose_real_score"].mean().item()
        nose_fake_score_val = loss_dict["nose_fake_score"].mean().item()
        mouth_real_score_val = loss_dict["mouth_real_score"].mean().item()
        mouth_fake_score_val = loss_dict["mouth_fake_score"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            # R1_loss=r1_val, Path_loss=path_loss_val, mean_path=mean_path_length_avg, augment=ada_aug_p)
            writer.add_scalar('G_Loss/Epoch', g_loss_val, i)
            writer.add_scalar('D_Face_Loss/Epoch', d_face_loss_val, i)
            writer.add_scalar('D_Eyes_Loss/Epoch', d_eyes_loss_val, i)
            writer.add_scalar('D_Nose_Loss/Epoch', d_nose_loss_val, i)
            writer.add_scalar('D_Mouth_Loss/Epoch', d_mouth_loss_val, i)
            writer.add_scalar('D_Loss/Epoch', d_loss_val, i)
            writer.add_scalar('R1_Loss/Epoch', r1_val, i)
            writer.add_scalar('Path_Loss/Epoch', path_loss_val, i)
            writer.add_scalar('Mean_Path', mean_path_length_avg, i)
            writer.add_scalar('Augment', ada_aug_p, i)

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            # Sampling images
            # if i % 100 == 0:
            #     with torch.no_grad():
            #         g_ema.eval()
            #         sample, _ = g_ema([sample_z])
                    # utils.save_image(
                    #     sample,
                    #     f"{save_dir}/{str(i).zfill(6)}.png",
                    #     nrow=int(args.n_sample ** 0.5),
                    #     normalize=True,
                    #     range=(-1, 1),
                    # )

            # Save pytorch checkpoint
            if i % 2000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d_face" : d_face_module.state_dict(),
                        "d_eyes" : d_eyes_module.state_dict(),        
                        "d_nose" : d_nose_module.state_dict(),
                        "d_mouth" : d_mouth_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                )

    writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument( "--n_sample",type=int,default=64,help="number of the samples generated during training",)
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize",type=float,default=2,help="weight of the path length regularization",)
    parser.add_argument("--path_batch_shrink",type=int,default=2,help="batch size reducing factor for the path length regularization (reduce memory consumption)",)
    parser.add_argument("--d_reg_every",type=int,default=16,help="interval of the applying r1 regularization",)
    parser.add_argument("--g_reg_every",type=int,default=4,help="interval of the applying path length regularization",)
    parser.add_argument("--mixing",type=float,default=0.9,help="probability of latent code mixing")
    parser.add_argument("--ckpt",type=str,default=None,help="path to the checkpoints to resume training",)
    parser.add_argument("--lr",type=float,default=0.002,help="learning rate")
    parser.add_argument("--channel_multiplier",type=int,default=2,help="channel multiplier factor for the model. config-f = 2, else = 1",)
    parser.add_argument("--wandb",action="store_true",help="use weights and biases logging")
    parser.add_argument("--local_rank",type=int,default=0,help="local rank for distributed training")
    parser.add_argument("--augment",action="store_true",help="apply non leaking augmentation")
    parser.add_argument("--augment_p",type=float,default=0,help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--ada_target",type=float,default=0.6,help="target augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_length",type=int,default=500 * 1000,help="target during to reach augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_every",type=int,default=256,help="probability update interval of the adaptive augmentation",)
    parser.add_argument("--freezeD", type=int, help="number of freezeD layers",default=-1)
    parser.add_argument("--freezeG", type=int, help="number of freezeG layers",default=-1)
    parser.add_argument("--structure_loss", type=int, help="number of structure loss layers",default=-1)
    parser.add_argument("--freezeStyle", type=int, help="freezeStyle",default=-1)
    parser.add_argument("--freezeFC", action="store_true",help="freezeFC",default=False)
    parser.add_argument("--gpu",default='cuda')
    parser.add_argument("--layerSwap",type=int,default=0)
    parser.add_argument("--expr_dir",default='expr')
    parser.add_argument('--source_impt',type=int,default=1)
    parser.add_argument('--adam_weight_decay',type=float,default=0.1)

    args = parser.parse_args()
    device = args.gpu

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator, Feature_Discriminator

    #----------------------------
    # Make Model
    #----------------------------
     
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    generator_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    feature_discriminators = {
        'face': Feature_Discriminator(args.size),
        'eyes': Feature_Discriminator(args.size),
        'nose': Feature_Discriminator(args.size),
        'mouth': Feature_Discriminator(args.size)
    }

    feature_discriminators['face'].to(device)
    feature_discriminators['eyes'].to(device)
    feature_discriminators['nose'].to(device)
    feature_discriminators['mouth'].to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.AdamW(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        weight_decay = args.adam_weight_decay,
    )
    
    d_optim = optim.AdamW(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        weight_decay = args.adam_weight_decay,
    )

    #----------------------------
    # Transfer Learning
    #----------------------------

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict = False)
        discriminator.load_state_dict(ckpt["d"], strict = False)
        g_ema.load_state_dict(ckpt["g_ema"], strict = False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        if 'd_face' in ckpt and 'd_eyes' in ckpt and 'd_nose' in ckpt and 'd_mouth' in ckpt:
            feature_discriminators['face'].load_state_dict(ckpt['d_face'], strict = False)
            feature_discriminators['eyes'].load_state_dict(ckpt['d_eyes'], strict = False)
            feature_discriminators['nose'].load_state_dict(ckpt['d_nose'], strict = False)
            feature_discriminators['mouth'].load_state_dict(ckpt['d_mouth'], strict = False)


    #----------------------------
    # GPU Setting
    #----------------------------

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )


    #----------------------------
    # Load the Data
    #----------------------------

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")


    #----------------------------
    # Train !
    #----------------------------

    train(args, loader, generator, generator_source, discriminator, g_optim, d_optim, g_ema, device, feature_discriminators)
