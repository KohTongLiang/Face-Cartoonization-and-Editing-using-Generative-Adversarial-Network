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

python generate-image.py

## Reference

- [happy-jihye/Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
