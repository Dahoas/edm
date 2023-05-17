torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=church_uno_out --seeds=0-10 --batch=8 \
    --network lsun_church_training-runs/uno.pkl/network-snapshot-008781.pkl \
    --img_resolution 128