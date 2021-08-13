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

def image2image():
    print("Loading Config")

    # load config file
    config = json.load(open('./config/config.json'))

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
    network1 = torch.load(network1)

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    for target in target_networks:
        print(f'Creating generator for {target}')

        network = target
        network = f'./networks/{network}.pt'
        network = torch.load(network)

        g = Generator(256, 512, 8, channel_multiplier=2).to(device)
        g.load_state_dict(network['g_ema'], strict=False)
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

        for i in range(number_of_img):
            # latent1

            latent2 = torch.randn(1, 14, 512, device=device)
            latent2 = g1.get_latent(latent2)
            img_gens = []

            for j in range(number_of_step):

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.45,
                                        return_latents=False,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=False,
                                        generator_name=f'ffhq-img-{i}-step-{j}')
                img_gens.append(imgs_gen1)
                for gen in target_generators:
                    imgs_gen, _ = gen['gen']([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.45,
                                        truncation_latent=gen['trunc'],
                                        randomize_noise=False,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        generator_name=f'{gen["gen_name"]}-img-{i}-step-{j}',
                                        )
                    img_gens.append(imgs_gen)          
            grid = make_grid(torch.cat(img_gens, 0),
                                nrow=no_of_networks+1,
                                normalize=True,
                                range=(-1,1),
                                )
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
            im.save(f'./asset/{outdir}/out-{i*number_of_step+j}.png')

            latent1 = latent2

        print('Complete')

def style_mixing():
    print("Loading Config")

    # load config file
    config = json.load(open('./config/config.json'))

    device = 'cuda'
    Target_network = "256Anime_332000" #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    swap = False #@param {type:"boolean"}
    save_swap_layer = 2 #@param {type:"slider", min:1, max:6, step:1}

    # ---------------
    # Generator
    # ---------------

    # Generator1
    # network1='./networks/ffhq256.pt' 
    # network1 = torch.load(network1)

    # generator1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    # generator1.load_state_dict(network1["g_ema"], strict=False)

    # trunc1 = generator1.mean_latent(4096)

    # latent1
    # seed1 = 827345 #@param {type:"slider", min:0, max:1000000, step:1}
    # torch.manual_seed(seed1)
    # latent1 = torch.randn(1, 14, 512, device=device)
    # latent1 = generator1.get_latent(latent1)

    # Generator2
    # network2=f'./networks/{Target_network}.pt' 
    # network2 = torch.load(network2)

    # generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    # generator2.load_state_dict(network2["g_ema"], strict=False)

    # trunc2 = generator2.mean_latent(4096)

    # latent2
    # seed2 = 309485 #@param {type:"slider", min:0, max:1000000, step:1}
    # torch.manual_seed(seed2)
    # latent2 = torch.randn(1, 14, 512, device=device)
    # latent2 = generator2.get_latent(latent2)

    # ---------------
    # Interpolation
    # ---------------

    number_of_step = 6 #@param {type:"slider", min:0, max:10, step:1}
    number_of_sample = 6
    swap_layer_num = 2

    outdir = config['outdir']
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./asset/{outdir}-style-mixing', exist_ok=True)
    # latent_interp = torch.zeros(number_of_step, latent1.shape[1], latent1.shape[2]).to(device)

    img_gens = []
    img = []

    with torch.no_grad():
        # for j in range(number_of_step):
        #     latent_interp[j] = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

        #     imgs_gen1, save_extract_layer = generator1([latent_interp],
        #                             input_is_latent=True,                                     
        #                             truncation=0.7,
        #                             truncation_latent=trunc1,
        #                             swap=swap, swap_layer_num=swap_layer_num,
        #                             )
                                    
        #     imgs_gen2, _ = generator2([latent_interp],
        #                             input_is_latent=True,                                     
        #                             truncation=0.7,
        #                             truncation_latent=trunc2,
        #                             swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
        #                             )

        # Generator1
        network1='./networks/256Anime_430000.pt' 
        network1 = torch.load(network1)

        generator1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
        generator1.load_state_dict(network1["g_ema"], strict=False)

        for i in range(number_of_sample):
            # latent1
            # seed1 = randrange(1000000) #@param {type:"slider", min:0, max:1000000, step:1}
            # torch.manual_seed(seed1)
            latent1 = torch.randn(1, 14, 512, device=device)
            latent1 = generator1.get_latent(latent1)
            trunc1 = generator1.mean_latent(4096)

            img_sample, latent = generator1([latent1], input_is_latent=True, return_latents=True)
            img_gens.append({ 'image' : img_sample, 'latent' : latent, 'trunc' : trunc1 })
            img.append(img_sample)

        # latent_interp = torch.zeros(number_of_step, latent1.shape[1], latent1.shape[2]).to(device)
        for style in img_gens:
            i = 0
            latent1 = style['latent']
            for style_to_change in img_gens:
                latent2 = style_to_change['latent']
                
                #mixed_latent = torch.tensor(latent2 , latent1)
                mixed, _ = generator1([latent1 + latent2], input_is_latent=True)
                img.append(mixed)
            
                i += 1

        grid = make_grid(torch.cat(img, 0),
                                    nrow=number_of_sample,
                                    normalize=True,
                                    range=(-1,1),
                                    )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = pilimg.fromarray(ndarr)
        im.save(f'./asset/{outdir}-style-mixing/result.png')

def sample_image():
    print("Loading Config")

    # load config file
    config = json.load(open('./config/config.json'))

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
        network = torch.load(network)

        g = Generator(256, 512, 8, channel_multiplier=2).to(device)
        g.load_state_dict(network['g_ema'], strict=False)
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
    swap = config['swap']
    swap_layer_num = config['swap_layer_num']
    # number_of_img = 4 #@param {type:"slider", min:0, max:30, step:1}
    # number_of_step = 6 #@param {type:"slider", min:0, max:10, step:1
    # swap = True #@param {type:"boolean"}
    # swap_layer_num = 2 #@param {type:"slider", min:1, max:6, step:1}

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
            im.save(f'./asset/{outdir}-sample-images/out-{i*number_of_step}.png')

        print('Complete')

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == '-i2i':
        image2image()
    elif mode == '-sample_image':
        sample_image()
    elif mode == '-style_mixing':
        style_mixing()