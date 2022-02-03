import torch
import numpy as np
import PIL.Image as pilimg

from torchvision.utils import save_image, make_grid
from model import Generator
from utils import imshow, tensor2image


if __name__ == "__main__":
    device = 'cuda'
    Target_network = "256Anime_200000" #@param ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Romance101', 'TrueBeauty', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG']
    swap = True #@param {type:"boolean"}
    save_swap_layer = 2 #@param {type:"slider", min:1, max:6, step:1}

    # ---------------
    # Generator
    # ---------------

    # Generator1
    network1='./networks/256Anime.pt' 
    network1 = torch.load(network1)

    generator1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator1.load_state_dict(network1["g_ema"], strict=False)

    trunc1 = generator1.mean_latent(4096)

    # latent1
    seed1 = 2734056 #@param {type:"slider", min:0, max:1000000, step:1}
    torch.manual_seed(seed1)
    latent1 = torch.randn(1, 14, 512, device=device)
    latent1 = generator1.get_latent(latent1)


    # Generator2
    network2=f'./networks/{Target_network}.pt' 
    network2 = torch.load(network2)

    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    generator2.load_state_dict(network2["g_ema"], strict=False)

    trunc2 = generator2.mean_latent(4096)

    # latent2
    seed2 = 145577 #@param {type:"slider", min:0, max:1000000, step:1}
    torch.manual_seed(seed2)
    latent2 = torch.randn(1, 14, 512, device=device)
    latent2 = generator2.get_latent(latent2)

    # ---------------
    # Interpolation
    # ---------------

    number_of_step = 3 #@param {type:"slider", min:0, max:10, step:1}
    latent_interp = torch.zeros(number_of_step, latent1.shape[1], latent1.shape[2]).to(device)
    swap_layer_num = 2 #@param {type:"slider", min:1, max:6, step:1}

    with torch.no_grad():
        for j in range(number_of_step):

            latent_interp[j] = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

            imgs_gen1, save_extract_layer = generator1([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.7,
                                    truncation_latent=trunc1,
                                    swap=swap, swap_layer_num=swap_layer_num,
                                    )

            print(latent_interp)
                                    
            imgs_gen2, _ = generator2([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.7,
                                    truncation_latent=trunc2,
                                    swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                    )

            grid = make_grid(torch.cat([imgs_gen1, imgs_gen2], 0),
                                    nrow=5,
                                    normalize=True,
                                    range=(-1,1),
                                    )
                
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = pilimg.fromarray(ndarr)
            im.save(f'./asset/result-style-mixing/out-{number_of_step+j}.png')
