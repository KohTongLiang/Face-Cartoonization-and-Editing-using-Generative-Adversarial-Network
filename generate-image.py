from math import trunc
import os
import torch
import argparse
import json
import cv2 
import sys
import numpy as np
import PIL.Image as pilimg

from torchvision.utils import save_image, make_grid
from utils import imshow, tensor2image
from model import Generator
from random import randrange

# init
# device=None
# truncation = None
# config_name = None
# seed = None
# index = None
# degree = None
# n_sample = None
# eigvec = None
# edit_image = True
# mode = ''
# config_name = ''
# swap = None
# swap_num = None


def image2image():
    print("Loading Config")

    # load config file
    config = json.load(open(f'./config/{config_name}.json'))

    # basic configuration
    device = config['cuda']
    truncation = 0.7

    # grab list of target networks
    target_networks = config['networks']
    target_generators = []
    no_of_networks = len(target_networks)

    # FFHQ Source Generator
    network1 = config['source_domain']  #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    network1 = f'./{network1}.pt' 
    network1 = torch.load(network1, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./{network}.pt'
        network = torch.load(network, map_location='cpu')

        g = Generator(256, 512, 8, channel_multiplier=2).to(device)
        g.load_state_dict(network['g_ema'], strict=False)
        g.to(device)
        trunc = g.mean_latent(4096)
        target_generators.append({ 'gen' : g, 'trunc' : trunc, 'gen_name' : target })

    # directory to save image
    # outdir = 'results_030821' #@param {type:"string"}
    outdir = config['outdir']
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./asset/{outdir}', exist_ok=True)

    imgs = []
    number_of_img = config['number_of_img']
    number_of_step = config['number_of_step']
    swap = config['swap']
    swap_layer_num = config['swap_layer_num']
    # number_of_img = 4 #@param {type:"slider", min:0, max:30, step:1}
    # number_of_step = 6 #@param {type:"slider", min:0, max:10, step:1
    # swap = True #@param {type:"boolean"}
    # swap_layer_num = 2 #@param {type:"slider", min:1, max:6, step:1}

    print("Beginning Generations")

    img_latents = []

    with torch.no_grad():

        latent1 = torch.randn(1, 14, 512, device=device)
        latent1 = g1.get_latent(latent1)
        latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)

        img_gens = []
        for i in range(number_of_img):
            # latent1

            latent2 = torch.randn(1, 14, 512, device=device)
            latent2 = g1.get_latent(latent2)

            for j in range(number_of_step):

                if number_of_step == 1:
                    latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step))
                else:
                    latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=config['truncation'],
                                        return_latents=False,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=True,
                                        generator_name=f'ffhq-img-{i}-step-{j}')
                img_gens.append(imgs_gen1)

                for gen in target_generators:
                    imgs_gen, img_latent = gen['gen']([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=config['truncation'],
                                        truncation_latent=gen['trunc'],
                                        randomize_noise=True,
                                        swap=swap,
                                        swap_layer_num=swap_layer_num,
                                        swap_layer_tensor=save_swap_layer,
                                        generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
                                        )
                    img_gens.append(imgs_gen)
                    img_latents.append(img_latent)
            latent1 = latent2
            
        grid = make_grid(torch.cat(img_gens, 0),
                            nrow=no_of_networks+1,
                            normalize=True,
                            range=(-1,1),
                            )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = pilimg.fromarray(ndarr)
        im.save(f'./asset/{outdir}/out.png')

        print('Complete')

        # print('Performing Sefa')

        # index=7
        # degree=15
        # img_gens = []
        # for deg in range(int(degree)):
        #     # direction = 0.5 * deg * eigvec[:, index].unsqueeze(0)
        #     # latent_interp += direction # image editing
        #     direction = 0.5 * deg * eigvec[:, index].unsqueeze(0)

        #     for i in img_latents:
        #         i += direction

        #         for gen in target_generators:
        #             imgs_gen, _ = gen['gen']([i],
        #                                 input_is_latent=True,                                     
        #                                 truncation=config['truncation'],
        #                                 truncation_latent=gen['trunc'],
        #                                 randomize_noise=True,
        #                                 swap=False,
        #                                 generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
        #                                 )
        #             img_gens.append(imgs_gen)

        # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # im = pilimg.fromarray(ndarr)
        # im.save(f'./asset/{outdir}/sefa.png')


def generate_image(latent=None, direction=None):
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
            latent1 = torch.randn(1, 14, 512, device=device)
            latent1 = g1.get_latent(latent1)
            latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)
            for i in range(number_of_img):
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
                                            generator_name=f'ffhq-img-{i}-step-{j}')
                    img_gens.append(imgs_gen1)
                    del imgs_gen1

                    for gen in target_generators:
                        imgs_gen, _ = gen['gen']([latent_interp],
                                            input_is_latent=True,                                     
                                            truncation=trunc_val,
                                            truncation_latent=gen['trunc'],
                                            randomize_noise=True,
                                            swap=swap,
                                            swap_layer_num=swap_layer_num,
                                            swap_layer_tensor=save_swap_layer,
                                            generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
                                            )
                        # gen['imgs'].append(imgs_gen)
                        img_gens.append(imgs_gen)
                latent1 = latent2
                
            grid = make_grid(torch.cat(img_gens, 0),
                                nrow=no_of_networks+1,
                                normalize=True,
                                range=(-1,1),
                                )
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
            im.save(f'./asset/{outdir}/out.png')
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FFHQ to Anime image translation and editing."
    )
    parser.add_argument("--gpu", type=str, default='cuda:1', help="Device id")
    parser.add_argument("--trunc", action="store_true", default=True, help="")
    parser.add_argument("--trunc_val", type=float, default=0.7, help="")
    parser.add_argument("--swap", action="store_true", default=True, help="")
    parser.add_argument("--swap_num", type=int, default=2, help="")
    parser.add_argument("--n_img", type=int, default=1, help="Number of sample image generated.")
    parser.add_argument("--n_step", type=int, default=1, help="Number of interpolation step.")
    parser.add_argument("--outdir", type=str, default='results', help="Output directory for images generated.")
    parser.add_argument("--config", type=str, default='c1', help="Config file with name of target networks.")
    parser.add_argument("--sefa", action="store_true", default=False, help="")
    parser.add_argument("--index", type=int, default=2, help="Index of eigen vector.")
    parser.add_argument("--degree", type=int, default=15, help="Degree of directions.")
    parser.add_argument("--n_sample", type=int, default=1, help=".")
    parser.add_argument("--factor", type=str, default='./networks/factor.pt', help=".")
    parser.add_argument("--seed", type=int, default=None, help=".")

    args = parser.parse_args()

    config_name = args.config
    device = args.gpu
    trunc = args.trunc
    trunc_val = args.trunc_val
    seed = args.seed
    index = args.index
    degree = args.degree
    n_sample = args.n_sample
    eigvec = torch.load(args.factor)["eigvec"].to(device)
    edit_image = args.sefa

    # load config file
    config = json.load(open(f'./config/{config_name}.json'))

    # grab list of target networks
    target_networks = config['networks']
    target_generators = []
    no_of_networks = len(target_networks)

    # FFHQ Source Generator
    network1 = config['source_domain']  #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    network1 = f'./{network1}.pt' 
    network1 = torch.load(network1, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./{network}.pt'
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

    print("Beginning Generations")

    if edit_image is True:
        latent = torch.randn(1, 14, 512, device=device)
        latent = g1.get_latent(latent)
        
        # latent1 = torch.randn(1, 14, 512, device=device)
        # latent1 = g1.get_latent(latent1)
        images = []
        i = 0
        for deg in range(int(degree)):
            direction = 0.5 * deg * eigvec[:, index].unsqueeze(0)

            im = generate_image(latent, direction)
            im.save(f'./asset/{outdir}/edited-{i}.png')
            images.append(im)
            i += 1
        images[0].save(f'./asset/{outdir}/sefa_result.gif', save_all=True, append_images=images[1:], loop=0, duration=100)
    else:
        im = generate_image()
        im.save(f'./asset/{outdir}/result.png')
    
    print('Complete, end of program.')