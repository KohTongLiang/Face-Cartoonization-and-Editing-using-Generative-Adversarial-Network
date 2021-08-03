import os
import torch
import json
import cv2 
import numpy as np
import PIL.Image as pilimg
from torchvision.utils import save_image, make_grid
from utils import imshow, tensor2image

from model import Generator

if __name__ == "__main__":
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

    # FFHQ Source Generator
    network1 = 'ffhq256'  #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
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
        target_generators.append({ 'gen' : g, 'trunc' : trunc })

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

            for j in range(number_of_step):
                img_gens = []

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=False,
                                        generator_name=f'ffhq-img-{i}-step-{j}')

                img_gens.append(imgs_gen1)

                for gen in target_generators:
                    imgs_gen, _ = gen['gen']([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=gen['trunc'],
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        generator_name=f'Gen2-img-{i}-step-{j}',
                                        )
                    img_gens.append(imgs_gen)
                                
                grid = make_grid(torch.cat(img_gens, 0),
                                    nrow=5,
                                    normalize=True,
                                    range=(-1,1),
                                    )
                
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = pilimg.fromarray(ndarr)
                im.save(f'./asset/{outdir}/out-{i*number_of_step+j}.png')

            latent1 = latent2

        print('Complete')
