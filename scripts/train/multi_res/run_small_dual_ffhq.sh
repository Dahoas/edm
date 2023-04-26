torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d datasets/ffhq-96x96.zip -d datasets/ffhq-80x80.zip -d datasets/ffhq-64x64.zip -d datasets/ffhq-48x48.zip -d datasets/ffhq-32x32.zip \
     --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/small_ffhq.yml \
    --lr 5e-4 --tick 25 --snap 100 --dump 100 \
    -dw 0.4 -dw 0.3 -dw 0.15 -dw 0.1 -dw 0.05 \
    --resume ffhq-training-runs/00109-ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/training-state-007526.pt