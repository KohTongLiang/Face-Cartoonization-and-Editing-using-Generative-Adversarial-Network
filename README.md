## Setting up config

Modify config file in /config/config.json:
{
    "cuda" : "cuda",                 // device id
    "outdir" : "results_040821",     // outdput directory where output images will be found
    "number_of_img" : 4,             // number of image to generate
    "number_of_step" : 6,            // number of step to take for each interpolation
    "swap" : true,                   // layer swapping
    "swap_layer_num" : 2,            // which layer to perform layer swapping on
    "networks" : [                   // list of networks to test
        "NaverWebtoon_StructureLoss",
        "Danbooru_Structure_Loss4",
        "Custom_NaverWebtoon"
    ]
}

## Running the program

python generate-image.py

## Reference

- [happy-jihye/Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
