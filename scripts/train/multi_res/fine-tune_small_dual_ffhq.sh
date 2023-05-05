torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d datasets/ffhq-96x96-to-160x160 -d datasets/ffhq-96x96.zip -d datasets/ffhq-80x80.zip -d datasets/ffhq-64x64.zip -d datasets/ffhq-48x48.zip -d datasets/ffhq-32x32.zip \
    -dw 0.3  -dw 0.14 -dw 0.14 -dw 0.14 -dw 0.14 -dw 0.14 \
    -bg 8 -bg 32 -bg 32 -bg 32 -bg 32 -bg 32 \
    --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/small_ffhq.yml \
    --lr 1e-4 --tick 25 --snap 50 --dump 50 \
    --resume ffhq-training-runs/00123-ffhq-96x96-to-160x160_ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/training-state-001254.pt