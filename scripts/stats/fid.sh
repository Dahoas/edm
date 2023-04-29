images=$1
ref=$2
torchrun --standalone --nproc_per_node=1 fid.py calc --images=$1 \
    --ref=$2