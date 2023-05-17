# Dual-FNO UNet: scale-robust diffusion model for zero-shot super-resolution image generation

Use this repo to train and sample DFU models. To train a DFU model run
```bash
bash scripts/train/multi_res/run_dual_ffhq.sh
```

To sample the trained DFU model run
```bash
bash scripts/infer/infer.sh
```

This repo is built on top of the [EDM repo](https://github.com/NVlabs/edm) from NVIDIA.
