## Trained Models

FFHQ: https://drive.google.com/file/d/1NrIQSs0DLeBVMOegg7QUHb08qFgDQy8_/view?usp=sharing

NaverWebtoon Structure Loss: https://drive.google.com/file/d/1kJ39AW6k_dlGYdh69ntOE7Jjj2ijivV_/view?usp=sharing

Anime Structure Loss (06-08-2021): https://drive.google.com/file/d/18AOOhYBHjmqyrEUNlIeMxfkQsh4HR927/view?usp=sharing

Store the .pt files in /networks

## Setting up config

Modify config file in /config/config.json
- cuda: device to run on
- outdir: outdput directory where output images will be found
- number_of_img: number of image to generate
- number_of_step: number of step to take for each interpolation
- swap: layer swapping
- swap_layer_num: which layer to perform layer swapping on
- networks: target networks to use for image generation
e.g) ["NaverWebtoon_StructureLoss", "Danbooru_Structure_Loss4", "Custom_NaverWebtoon"]


## Running the program

### Image generation
1) setup config file
2) run: python generate-image.py

## Training

### Layer Swapping
### ex) python train.py --batch=8 --ckpt=ffhq256.pt --layerSwap=2 --freezeD=3 --augment --path=LMDB_PATH

### StyleGAN2
python train.py --batch BATCH_SIZE LMDB_PATH
### ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH

### StructureLoss
### ex) python train.py --batch=8 --ckpt=ffhq256.pt --structure_loss=2 --freezeD=3 --augment --path=LMDB_PATH

### FreezeSG
### ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeStyle=2 --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH


### Distributed Settings
python train.py --batch BATCH_SIZE --path LMDB_PATH \
    -m torch.distributed.launch --nproc_per_node=N_GPU --main_port=PORT


## Reference

- [happy-jihye/Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
