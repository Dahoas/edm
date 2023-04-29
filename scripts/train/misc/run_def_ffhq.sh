torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    --data=datasets/ffhq-96x96.zip --arch=ddpmpp --mode="def" --batch 256 \
    --model_config_path model_configs/small_ffhq.yml \
    --lr 1e-3 --tick 25 --snap 10