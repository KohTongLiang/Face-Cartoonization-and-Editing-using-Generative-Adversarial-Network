from math import trunc
import os
import torch
import json
import cv2 
import sys
import numpy as np
import PIL.Image as pilimg

from torchvision.utils import save_image, make_grid
from utils import imshow, tensor2image
from model import Generator
from random import randrange

mode = ''
config_name = ''


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
    network1 = f'./networks/{network1}.pt' 
    network1 = torch.load(network1, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./networks/{network}.pt'
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

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.5,
                                        return_latents=False,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=True,
                                        generator_name=f'ffhq-img-{i}-step-{j}')
                img_gens.append(imgs_gen1)
                for gen in target_generators:
                    imgs_gen, _ = gen['gen']([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.5,
                                        truncation_latent=gen['trunc'],
                                        randomize_noise=True,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
                                        )
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

        print('Complete')

def style_mixing():
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
    network1 = f'./networks/{network1}.pt' 
    network1 = torch.load(network1, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)
    print(f'Running on cuda:{torch.cuda.current_device()}')

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./networks/{network}.pt'
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
        os.makedirs(f'./asset/{outdir}-style-mixing', exist_ok=True)

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

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, returned_latent = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        return_latents=True,
                                        truncation_latent=trunc1,
                                        randomize_noise=False,
                                        generator_name=f'ffhq-img-{i}-step-{j}')
                img_gens.append(imgs_gen1)
                for gen in target_generators:
                    imgs_gen, _ = gen['gen']([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=gen['trunc'],
                                        inject_index=config['inject_index'],
                                        randomize_noise=True,
                                        put_latent=returned_latent,
                                        generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
                                        )
                    img_gens.append(imgs_gen)
            
            latent1 = latent2

        grid = make_grid(torch.cat(img_gens, 0),
                                nrow=no_of_networks+1,
                                normalize=True,
                                range=(-1,1),
                                )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = pilimg.fromarray(ndarr)
        im.save(f'./asset/{outdir}-style-mixing/out.png')

        print('Complete')

def sample_image():
    print("Loading Config")

    # load config file
    config = json.load(open(f'./config/{config_name}.json'))

    # basic configuration
    device = config['cuda']
    n_sample = 5
    truncation = 0.7

    # grab list of target networks
    target_networks = config['networks']
    target_generators = []
    no_of_networks = len(target_networks)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./networks/{network}.pt'
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
        os.makedirs(f'./asset/{outdir}-sample-images', exist_ok=True)

    imgs = []
    number_of_img = config['number_of_img']
    number_of_step = config['number_of_step']
    print("Beginning Generations")

    with torch.no_grad():
        img_gens = []
        for i in range(number_of_img):
            for gen in target_generators:
                latent = torch.randn(1, 14, 512, device=device)
                latent = gen['gen'].get_latent(latent)

                imgs_gen, _ = gen['gen']([latent],
                                    input_is_latent=True,                           
                                    truncation=0.7,
                                    truncation_latent=gen['trunc'],
                                    generator_name=f'{gen["gen_name"]}-img-{i}',
                                    )
                img_gens.append(imgs_gen)

        grid = make_grid(torch.cat(img_gens, 0),
                            nrow=no_of_networks,
                            normalize=True,
                            range=(-1,1),
                            )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = pilimg.fromarray(ndarr)
        im.save(f'./asset/{outdir}-sample-images/out.png')

        print('Complete')

if __name__ == "__main__":
    mode = sys.argv[1]
    config_name = sys.argv[2]

    if mode == 'i2i':
        image2image()
    elif mode == 'sample_image':
        sample_image()
    elif mode == 'style_mixing':
        style_mixing()