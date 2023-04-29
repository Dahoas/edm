torchrun --standalone --nproc_per_node=8 train.py --outdir=lsun-training-runs \
    --data=datasets/lsun/lsun_chuch_128.zip --arch=ddpmpp --mode="dual" --batch 128 \
    --fp16 True --model_config_path model_configs/lsun.yml