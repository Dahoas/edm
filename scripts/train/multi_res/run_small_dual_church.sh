torchrun --standalone --nproc_per_node=8 train.py --outdir=lsun_church_training-runs \
    -d datasets/lsun_church_96x96 -d datasets/lsun_church_80x80 -d datasets/lsun_church_64x64 -d datasets/lsun_church_48x48 -d datasets/lsun_church_32x32 \
     --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/small_church.yml \
    --lr 1e-4 --tick 25 --snap 25 \
    --transfer lsun_church_training-runs/uno.pkl/network-snapshot-008781.pkl