torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-0 --batch=64 \
    --network ffhq-training-runs/network-snapshot-073382.pkl \
    --img_resolution 128