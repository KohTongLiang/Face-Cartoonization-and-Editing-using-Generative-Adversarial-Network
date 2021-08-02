import os
import torch
import cv2 
import numpy as np
import PIL.Image as pilimg
from torchvision.utils import save_image, make_grid
from utils import imshow, tensor2image

from model import Generator

if __name__ == "__main__":
    #torch.cuda.set_device(1)
    device='cuda'
    n_sample=5
    truncation = 0.7

    # =============================================

    # FFHQ Source Generator
    network1 = 'ffhq256'  #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    network1 = f'./networks/{network1}.pt' 
    network1 = torch.load(network1)

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    trunc1 = g1.mean_latent(4096)

    # NaverWebtoon Structure Loss
    network3 = 'NaverWebtoon_StructureLoss' #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    network3 = f'./networks/{network3}.pt'
    network3 = torch.load(network3)

    g3 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g3.load_state_dict(network3["g_ema"], strict=False)
    trunc3 = g3.mean_latent(4096)

    # Danbooru trained with 2* relative importance of source domain
    network5 = '100000'
    network5 = f'./expr/checkpoints/{network5}.pt'
    network5 = torch.load(network5)

    g5 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g5.load_state_dict(network5['g_ema'], strict=False)
    trunc5 = g5.mean_latent(4096)

    # Kaggle Structure Loss
    network7 = 'Custom_NaverWebtoon'
    network7 = f'./networks/{network7}.pt'
    network7 = torch.load(network7)

    g7 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g7.load_state_dict(network7['g_ema'], strict=False)
    trunc7 = g7.mean_latent(4096)

    # NaverWebtoon_FreezeSG
    network8 = 'Danbooru_FreezeSG'
    network8 = f'./networks/{network8}.pt'
    network8 = torch.load(network8)

    g8 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g8.load_state_dict(network8['g_ema'], strict=False)
    trunc8 = g8.mean_latent(4096)

    # =============================================

    # directory to save image
    outdir = 'results_020821_noswap' #@param {type:"string"}
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./asset/{outdir}', exist_ok=True)

    imgs = []
    number_of_img = 4 #@param {type:"slider", min:0, max:30, step:1}
    number_of_step = 6 #@param {type:"slider", min:0, max:10, step:1
    swap = True #@param {type:"boolean"}
    swap_layer_num = 2 #@param {type:"slider", min:1, max:6, step:1}

    with torch.no_grad():

        latent1 = torch.randn(1, 14, 512, device=device)
        latent1 = g1.get_latent(latent1)
        latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)

        for i in range(number_of_img):
            # latent1

            latent2 = torch.randn(1, 14, 512, device=device)
            latent2 = g1.get_latent(latent2)

            for j in range(number_of_step):

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=False)
                imgs_gen3, _ = g3([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc3,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                imgs_gen5, _ = g5([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc5,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                imgs_gen7, _ = g7([latent_interp],
                                        input_is_latent=True,
                                        truncation=0.7,
                                        truncation_latent=trunc7,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                imgs_gen8, _ = g8([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc8,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                                
                grid = make_grid(torch.cat([imgs_gen1, imgs_gen3, imgs_gen5, imgs_gen7, imgs_gen8], 0),
                                    nrow=5,
                                    normalize=True,
                                    range=(-1,1),
                                    )
                
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = pilimg.fromarray(ndarr)
                im.save(f'./asset/{outdir}/out-{i*number_of_step+j}.png')

            latent1 = latent2

        print('done')
