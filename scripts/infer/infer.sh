torchrun --standalone --nnodes=1 --nproc_per_node=1 generate.py --outdir=lsun_def_out/down/ --seeds=0-10 --batch=64 \
    --network lsun-training-runs/00003-lsun_church_96x96-uncond-ddpmpp-edm-gpus8-batch256-fp16/network-snapshot-056448.pkl \
    --img_resolution 32