import os
import torch
import argparse
import PIL.Image as pilimg

from torchvision.utils import make_grid
from model import Generator

def main():    
    parser = argparse.ArgumentParser(description="FFHQ to Anime image translation.")

    # environment
    parser.add_argument("--outdir", type=str, default='results', help="Output directory for images generated.")
    parser.add_argument("--gpu", type=str, default='cuda', help="Device id")
    parser.add_argument("--output_name", type=str, default='result')

    # parameters
    parser.add_argument("--trunc", action="store_true", default=True, help="")
    parser.add_argument("--size", type=int, default=256, help="output image sizes of the generator")
    parser.add_argument("--trunc_val", type=float, default=0.7, help="")
    parser.add_argument("--swap", action="store_true", default=False, help="")
    parser.add_argument("--swap_num", type=int, default=2, help="")
    parser.add_argument("--n_img", type=int, default=1, help="Number of sample image generated.")
    parser.add_argument("--n_step", type=int, default=1, help="Number of interpolation step.")
    parser.add_argument("--n_sample", type=int, default=1, help=".")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed for reprocudibility.")

    # models
    parser.add_argument("--e_ckpt", type=str, default='./networks/encoder_ffhq_encoder_200000.pt', help="path to the encoder checkpoint")
    parser.add_argument("--source", type=str, default='networks/ffhq256.pt', help="File path and name to source generator network.")
    parser.add_argument("--target", type=str, default='networks/tl_fyp.pt', help="File path and name for all target networks. Add multiple like network2,network3,network4.")

    args = parser.parse_args()

    device = args.gpu
    trunc = args.trunc
    trunc_val = args.trunc_val
    source = args.source
    seed = args.seed
    target = args.target
    number_of_img = args.n_img
    number_of_step = args.n_step
    swap_layer_num = args.swap_num
    swap = args.swap
    outdir = args.outdir
    output_name = args.output_name

    img_gens = []
    
    target_networks = target.split(',')
    # grab list of target networks
    target_generators = []
    no_of_networks = len(target_networks)

    # FFHQ Source Generator
    source = f'./{source}' 
    source = torch.load(source, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(source["g_ema"], strict=False)
    g1.to(device)
    trunc1 = g1.mean_latent(4096)

    # Create generator of each target networks
    print("Printing " + str(target_networks))
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
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./{outdir}', exist_ok=True)

    # apply manual seed
    if seed is not None:
        torch.manual_seed(seed)

    print("Beginning Program")
    latent1, latent2  = None, None

    with torch.no_grad():
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
            im.save(f'./{outdir}/{output_name}.png')
    
    print('Complete. Images generated.')

if __name__ == '__main__':
    main()