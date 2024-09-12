CUDA_VISIBLE_DEVICES=0,1 python -u main_swap.py \
--logdir models/REFace/ \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/train.yaml \
--scale_lr False 


