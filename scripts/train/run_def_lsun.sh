torchrun --standalone --nproc_per_node=8 train.py --outdir=lsun_church_training-runs \
    --data=datasets/lsun_church_96x96 --arch=ddpmpp --mode="def" --batch 64 \
    --model_config_path model_configs/ffhq.yml \
    --lr 1e-3 --tick 25 --snap 10