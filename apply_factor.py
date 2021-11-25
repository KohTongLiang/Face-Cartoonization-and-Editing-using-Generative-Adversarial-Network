import os
import argparse
import torch
from torchvision import utils
import PIL.Image as pilimg
from torchvision.utils import make_grid
from model import Generator

def make_video(args):
    # Eigen-Vector
    eigvec = torch.load(args.factor)["eigvec"].to(args.device)

    # =============================================

    # Generator 1
    network1 = torch.load(args.ckpt)

    g1 = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    g1.to(args.device)
    trunc1 = g1.mean_latent(4096)

    # Generator2
    # Target generator
    network2 = torch.load(args.ckpt2)

    g2 = Generator(256, 512, 8, channel_multiplier=2).to(args.device)
    g2.load_state_dict(network2["g_ema"], strict=False)
    g2.to(args.device)
    trunc2 = g2.mean_latent(4096)

    # latent
    if args.seed is not None:
        torch.manual_seed(args.seed)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g1.get_latent(latent)

    # latent direction & scalar
    index=args.index
    degree=args.degree

    # =============================================
    
    images = []

    for deg in range(int(degree)):

        direction = 0.5 * deg * eigvec[:, index].unsqueeze(0)

        img1, save_swap_layer = g1(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc1,
            input_is_latent=True,
            return_latents=False,
            swap=True,
            swap_layer_num=2,
            randomize_noise=True,
        )

        img2, _ = g2(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc2,
            input_is_latent=True,
            randomize_noise=True,
            swap=True,
            swap_layer_num=2,
            swap_layer_tensor=save_swap_layer,
        )

        grid = make_grid(torch.cat([img1, img2], 0),
                        nrow=args.n_sample,
                        normalize=True,
                        range=(-1,1),
                        )
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = pilimg.fromarray(ndarr)
        images.append(img)

    images[0].save(f'{args.outdir}/{args.video_name}.gif', save_all=True, append_images=images[1:], loop=0, duration=100)

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    # Make GIF
    parser.add_argument(
        "--video",
        action="store_true",
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--video_name",
        type=str, default='video',
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--ckpt2", type=str, help="If you make a video, enter the required stylegan2 checkpoints for transfer learning")
    parser.add_argument("--seed", type=int, default=None)
    

    args = parser.parse_args()

    # =============================================

    # directory to save image
    if not os.path.isdir(f'{args.outdir}'):
        os.makedirs(f'{args.outdir}')


    # Eigen-Vector
    eigvec = torch.load(args.factor)["eigvec"].to(args.device)

    # Generator
    ckpt = torch.load(args.ckpt, map_location='cpu')
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    # latent
    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    # direction
    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    grid = utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.outdir}/{args.video_name}.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )

    # make video
    if args.video == True:
        make_video(args)
