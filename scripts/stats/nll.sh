torchrun --standalone --nproc_per_node=1 nll.py --network DFU/network-snapshot-100352.pkl \
    --image_path DFU/ffhq-32x32-val.zip --sigma_min 0.002 --sigma_max 80 \
    --eval_dir DFU/32/