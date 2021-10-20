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

## Preparing Dataset
# for images in folder
python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

# for zip file
python run.py --prepare_data=DATASET_PATH --zip=ZIP_NAME --size SIZE

## Running the program

### Image generation (example)
1) setup config file
2) run: python generate-image.py mode config_file_name # mode can be i2i, sample_image, style_mixing


## Training

### StyleGAN2
python train.py --batch BATCH_SIZE LMDB_PATH

ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH --expr_dir=Experiment_Directory --gpu=CUDA:0

### StructureLoss
ex) python train.py --batch=8 --ckpt=ffhq256.pt --structure_loss=2 --freezeD=3 --augment --path=LMDB_PATH --expr_dir=Experiment_Directory --gpu=CUDA:0

### FreezeSG
ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeStyle=2 --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH --expr_dir=Experiment_Directory --gpu=CUDA:0

## Evaluation

FID: python -m pytorch_fid ./datasetA ./datasetB

## Reference

- [happy-jihye/Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
