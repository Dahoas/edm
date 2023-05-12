torchrun --standalone --nproc_per_node=8 train.py --outdir=lsun-training-runs \
    -d datasets/lsun_church_96x96 -d datasets/lsun_church_80x80 -d datasets/lsun_church_64x64 -d datasets/lsun_church_48x48 -d datasets/lsun_church_32x32 \
     --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/small_ffhq.yml \
    --lr 5e-4 --tick 25 --snap 100 --dump 100 