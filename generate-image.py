from ctypes import Structure
from math import trunc
import os
from numpy.lib.utils import source
import torch
import argparse
import json
import cv2 
import sys
import numpy as np
import PIL.Image as pilimg
import lpips

from torchvision.utils import save_image, make_grid
from utils import imshow, tensor2image
from model import Generator, Encoder
from random import randrange
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from utils import tensor2image, save_image
from tqdm import tqdm


# perform image generation or editing
def generate_image(latent=None, direction=None, latent1=None, latent2=None):
    img_gens = []

    with torch.no_grad():        
        if direction is not None and latent is not None:
            img1, save_swap_layer = g1(
                [latent + direction],
                truncation=True,
                truncation_latent=trunc1,
                input_is_latent=True,
                return_latents=False,
                swap=True,
                swap_layer_num=2,
                randomize_noise=True,
            )
            img_gens.append(img1)

            for gen in target_generators:
                imgs_gen, _ = gen['gen']([latent + direction],
                                    input_is_latent=True,                                     
                                    truncation=trunc_val,
                                    truncation_latent=gen['trunc'],
                                    randomize_noise=True,
                                    swap=swap,
                                    swap_layer_num=swap_layer_num,
                                    swap_layer_tensor=save_swap_layer,
                                    )
                img_gens.append(imgs_gen)

            grid = make_grid(torch.cat(img_gens, 0),
                            nrow=n_sample,
                            normalize=True,
                            range=(-1,1),
                            )
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
        else:
            if latent1 is None:
                latent1 = torch.randn(1, 14, 512, device=device)
                latent1 = g1.get_latent(latent1)
            latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)
            for i in range(number_of_img):
                # if latent2 is None:
                latent2 = torch.randn(1, 14, 512, device=device)
                latent2 = g1.get_latent(latent2)

                for j in range(number_of_step):
                    if number_of_step == 1:
                        latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step))
                    else:
                        latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                    imgs_gen1, save_swap_layer = g1([latent_interp],
                                            input_is_latent=True,                                     
                                            truncation=trunc_val,
                                            return_latents=False,
                                            truncation_latent=trunc1,
                                            swap=swap, swap_layer_num=swap_layer_num,
                                            randomize_noise=True,
                                            )
                    img_gens.append(imgs_gen1)
                    del imgs_gen1

                    target_im = 0
                    for gen in target_generators:
                        imgs_gen, _ = gen['gen']([latent_interp],
                                            input_is_latent=True,                                     
                                            truncation=trunc_val,
                                            truncation_latent=gen['trunc'],
                                            randomize_noise=True,
                                            swap=swap,
                                            swap_layer_num=swap_layer_num,
                                            swap_layer_tensor=save_swap_layer,
                                            )
                        # gen['imgs'].append(imgs_gen)
                        img_gens.append(imgs_gen)
                        target_im += 1
                latent1 = latent2
                
            grid = make_grid(torch.cat(img_gens, 0),
                                nrow=no_of_networks+1,
                                normalize=True,
                                range=(-1,1),
                                )
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
            im.save(f'./asset/{outdir}/out.png')
    return im, img_gens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFHQ to Anime image translation and editing.")

    parser.add_argument("--gpu", type=str, default='cuda:1', help="Device id")
    parser.add_argument("--trunc", action="store_true", default=True, help="")
    parser.add_argument("--size", type=int, default=256, help="output image sizes of the generator")
    parser.add_argument("--trunc_val", type=float, default=0.7, help="")
    parser.add_argument("--swap", action="store_true", default=False, help="")
    parser.add_argument("--swap_num", type=int, default=2, help="")
    parser.add_argument("--n_img", type=int, default=1, help="Number of sample image generated.")
    parser.add_argument("--n_step", type=int, default=1, help="Number of interpolation step.")
    parser.add_argument("--outdir", type=str, default='results', help="Output directory for images generated.")
    parser.add_argument("--sefa", action="store_true", default=False, help="Set this flag if you wish to perform image editing.")
    parser.add_argument("--index", type=str, default="2", help="Index of eigen vector. Passed in as 2,3,4 etc.")
    parser.add_argument("--degree", type=int, default=6, help="Degree of directions.")
    parser.add_argument("--n_sample", type=int, default=1, help=".")
    parser.add_argument("--factor", type=str, default='networks/tl_fyp_factor.pt', help=".")
    parser.add_argument("--files", nargs="+", help="path to image files to be projected")
    parser.add_argument("--e_ckpt", type=str, default='./networks/encoder_ffhq_encoder_200000.pt', help="path to the encoder checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed for reprocudibility.")
    parser.add_argument("--output_name", type=str, default='result')
    parser.add_argument("--sefa_name", type=str, default='sefa')
    parser.add_argument("--source", type=str, default='networks/ffhq256.pt', help="File path and name to source generator network.")
    parser.add_argument("--target", type=str, default='networks/tl_fyp.pt', help="File path and name for all target networks. Add multiple like network2,network3,network4.")
    parser.add_argument("--explore", type=int, default=0, help="Loop through a set of indices to explore semantic meaning for Sefa.")

    args = parser.parse_args()

    device = args.gpu
    trunc = args.trunc
    trunc_val = args.trunc_val
    index = args.index.split(',')
    degree = args.degree
    n_sample = args.n_sample
    edit_image = args.sefa
    network1 = args.source
    seed = args.seed
    target_networks = args.target.split(',')
    index = [int(n) for n in index]

    # grab list of target networks
    target_generators = []
    no_of_networks = len(target_networks)

    # FFHQ Source Generator
    network1 = f'./{network1}' 
    network1 = torch.load(network1, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./{network}'
        network = torch.load(network, map_location='cpu')

        g = Generator(256, 512, 8, channel_multiplier=2).to(device)
        g.load_state_dict(network['g_ema'], strict=False)
        g.to(device)
        trunc = g.mean_latent(4096)
        target_generators.append({ 'gen' : g, 'trunc' : trunc, 'gen_name' : target, 'imgs' : [] })

    # directory to save image
    outdir = args.outdir
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./asset/{outdir}', exist_ok=True)

    imgs = []
    number_of_img = args.n_img #@param {type:"slider", min:0, max:30, step:1}
    number_of_step = args.n_step #@param {type:"slider", min:0, max:10, step:1
    swap = args.swap #@param {type:"boolean"}
    swap_layer_num = args.swap_num #@param {type:"slider", min:1, max:6, step:1}

    # apply manual seed
    if seed is not None:
        torch.manual_seed(seed)

    print("Beginning Program")
    latent1, latent2  = None, None


    if edit_image is True:
        eigvec = torch.load(f'./{args.factor}')["eigvec"].to(device)
        if latent1 is None:
            latent1 = torch.randn(n_sample, 14, 512, device=device)
            latent1 = g1.get_latent(latent1)
               
        if args.explore > 0:
            for idx in range(index[-1], args.explore):
                # explore a range of indices
                print(f"Exploring index {idx}")
                images = []
                for deg in range(int(degree)):
                    direction = 0.5 * deg * eigvec[:, idx].unsqueeze(0)

                    im, _ = generate_image(latent1, direction)
                    images.append(im)
                images[0].save(f'./asset/{outdir}/explore-{args.sefa_name}-idx-{idx}.gif', save_all=True, append_images=images[1:], loop=0, duration=100)
        else: 
            i = 0
            images = []
            for deg in range(int(degree)):

                # concatenate multiple features
                # direction = 0.5 * deg * eigvec[:, index[0]].unsqueeze(0)
                direction = []
                for j in range(len(index)):
                    direction.append(0.5 * deg * eigvec[:, index[j]].unsqueeze(0))
                direction = sum(direction)

                im, imgs_gens = generate_image(latent1, direction)
                # im.save(f'./asset/{outdir}/{args.sefa_name}-{i}.png')
                images.append(im)
                i += 1
                grid = make_grid(torch.cat(imgs_gens, 0),
                            nrow=int(degree),
                            normalize=True,
                            range=(-1,1),
                            )
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
            im.save(f'./asset/{outdir}/{args.sefa_name}.png')
            images[0].save(f'./asset/{outdir}/{args.sefa_name}.gif', save_all=True, append_images=images[1:], loop=0, duration=100)
    else:
        im, _ = generate_image(latent1=latent1, latent2=latent2)
        im.save(f'./asset/{outdir}/{args.output_name}.png')
    
    print('Complete, end of program.')