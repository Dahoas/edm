# Dual-FNO UNet: scale-robust diffusion model for zero-shot super-resolution image generation

This repo is adapted from [EDM repo](https://github.com/NVlabs/edm) from NVIDIA, used under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License]{https://creativecommons.org/licenses/by-nc-sa/4.0/} by NVIDIA under Copyright © 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Our work "Dual-FNO UNet: scale-robust diffusion model for zero-shot super-resolution image generation" is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License]{https://creativecommons.org/licenses/by-nc-sa/4.0/}.

Use this repo to train and sample DFU models. 

## Training
To train a DFU model run
```bash
bash scripts/train/multi_res/run_dual.sh
```
Some of the relevant training options are:
```
--mode Architecture mode, options are 'def' (Unet), 'fourier' (FNO Unet), 'dual' (Dual FNO Unet)
-d --data Path to the dataset to train on. It can be a folder or a zip file. 
--model_config_path Path to the config yml. See ./model_configs for examples
```

For more options look at the train.py file. 

#### Fine tuning

For fine tunning run:
```bash
bash scripts/train/multi_res/fine-tune_small_dual_ffhq.sh
```
To freeze certain layers include them in your model_config.yml, for instance the following freezes the norm and spectral layers in the first and second block of the unet and every layer of the 0-th block:

```yml
frozen_layers:
  0: ["all"]
  1: ["norm", "spectral"]
  2: ["norm", "spectral"]
```

## Sampling
To sample from a trained model run the script
```bash
bash scripts/infer/infer.sh
```
Some of the relevant options are: 
```
--network Path to the score network
--img_resolution Resolution at which to sample from
```

For more options look at the generate.py file. 


## FID Computation
Given a set of images and a statistics reference compute fid by running: 

```bash
bash scripts/stats/fid.sh images_path stats_ref_path
```

To compute the statistics for a set of images use the script:

```bash
bash scripts/stats/fid.sh 
```

The relevant options here are:
```
--data Path to images, can be a folder or zip file
--dest File to save the reference statistics
```
