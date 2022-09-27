import os
import torch
import argparse
import PIL.Image as pilimg

from torchvision.utils import make_grid
from model import Generator
from utils import tensor2image

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
    parser.add_argument("--factor", type=str, default='networks/tl_fyp_factor.pt', help=".")

    # models
    parser.add_argument("--e_ckpt", type=str, default='./networks/encoder_ffhq_encoder_200000.pt', help="path to the encoder checkpoint")
    parser.add_argument("--source", type=str, default='networks/ffhq256.pt', help="File path and name to source generator network.")
    parser.add_argument("--target", type=str, default='networks/tl_fyp.pt', help="File path and name for all target networks. Add multiple like network2,network3,network4.")
    
    # Image editing
    parser.add_argument("--index", type=str, default="2", help="Index of eigen vector. Passed in as 2,3,4 etc.")
    parser.add_argument("--degree", type=int, default=6, help="Degree of directions.")
    parser.add_argument("--explore", type=int, default=0, help="Loop through a set of indices to explore semantic meaning for Sefa.")
    parser.add_argument("--sefa_name", type=str, default='sefa')

    args = parser.parse_args()

    device = args.gpu
    trunc = args.trunc
    trunc_val = args.trunc_val
    degree = args.degree
    n_sample = args.n_sample
    source = args.source
    seed = args.seed
    target=args.target
    swap_layer_num = args.swap_num
    swap = args.swap
    outdir = args.outdir
    sefa_name = args.sefa_name
    factor = args.factor

    index = args.index.split(',')
    index = [int(n) for n in index]

    img_gens = []
    
    target_networks = target.split(',')
    # grab list of target networks
    target_generators = []

    # FFHQ Source Generator
    source = f'./{source}' 
    source = torch.load(source, map_location='cpu')

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(source["g_ema"], strict=False)
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
    if not os.path.isdir(f'{outdir}'):
        os.makedirs(f'./{outdir}', exist_ok=True)

    # apply manual seed
    if seed is not None:
        torch.manual_seed(seed)

    print("Beginning Program")
    latent1, latent2  = None, None

    with torch.no_grad():
        eigvec = torch.load(f'./{factor}')["eigvec"].to(device)
        if latent1 is None:
            latent1 = torch.randn(n_sample, 14, 512, device=device)
            latent1 = g1.get_latent(latent1)
            
        # if args.explore > 0:
        #     for idx in range(index[-1], args.explore):
        #         # explore a range of indices
        #         print(f"Exploring index {idx}")
        #         images = []
        #         for deg in range(int(degree)):
        #             direction = 0.5 * deg * eigvec[:, idx].unsqueeze(0)

        #             im, _ = generate_image(latent1, direction)
        #             images.append(im)
        #         images[0].save(f'./{outdir}/explore-{args.sefa_name}-idx-{idx}.gif', save_all=True, append_images=images[1:], loop=0, duration=100)
        # else: 
        i = 0
        images = []
        for deg in range(int(degree)):

            # concatenate multiple features
            # direction = 0.5 * deg * eigvec[:, index[0]].unsqueeze(0)
            direction = []
            for j in range(len(index)):
                direction.append(0.5 * deg * eigvec[:, index[j]].unsqueeze(0))
            direction = sum(direction)

            if direction is not None and latent1 is not None:
                img1, save_swap_layer = g1(
                    [latent1 + direction],
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
                    imgs_gen, _ = gen['gen']([latent1 + direction],
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
            images.append(im)
            i += 1
            grid = make_grid(torch.cat(img_gens, 0),
                        nrow=int(degree),
                        normalize=True,
                        range=(-1,1),
                        )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = pilimg.fromarray(ndarr)
        im.save(f'./{outdir}/{sefa_name}.png')
        images[0].save(f'./{outdir}/{sefa_name}.gif', save_all=True, append_images=[i for i in images[1:]], loop=0, duration=100)

    print('Complete. Images edited.')

if __name__ == "__main__":
    main()