# 006
CUDA_VISIBLE_DEVICES=0

python main.py --config_dir configs/DIRG_020/config_NNSPN.yaml

CUDA_VISIBLE_DEVICES=6,7 python main.py --config_dir configs/DIRG_020/config_NNSPN_gen.yaml