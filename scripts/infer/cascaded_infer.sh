torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_cascaded_generation --seeds=0-63 --batch=64 \
    --network ffhq-training-runs/network-snapshot-073382.pkl \
    --img_resolution 32