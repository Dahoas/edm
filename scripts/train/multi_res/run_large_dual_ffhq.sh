torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d datasets/ffhq-128x128.zip -d datasets/ffhq-96x96.zip -d datasets/ffhq-64x64.zip -d datasets/ffhq-48x48.zip -d datasets/ffhq-32x32.zip \
     --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/large_ffhq.yml \
    --lr 1e-3 --tick 25 --snap 25