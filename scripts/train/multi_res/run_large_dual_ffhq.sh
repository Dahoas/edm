torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d datasets/ffhq-160x160.zip -d datasets/ffhq-144x144.zip -d datasets/ffhq-128x128.zip -d datasets/ffhq-112x112.zip -d datasets/ffhq-96x96.zip \
     --arch=ddpmpp --mode="dual" --batch 256  --batch-gpu 8 \
    --model_config_path model_configs/large_ffhq.yml \
    --lr 5e-5 --tick 25 --snap 75 --dump 75 \
    --resume ffhq-training-runs/00084-ffhq-160x160_ffhq-144x144_ffhq-128x128_ffhq-112x112_ffhq-96x96-uncond-ddpmpp-edm-gpus8-batch256-fp32/training-state-035123.pt
