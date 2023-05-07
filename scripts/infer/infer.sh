torchrun --standalone --nnodes=1 --nproc_per_node=1 generate.py --outdir=lsun_def_out/ --seeds=0-10 --batch=64 \
    --network lsun_church_training-runs/00001-____-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-040141.pkl \
    --img_resolution 192