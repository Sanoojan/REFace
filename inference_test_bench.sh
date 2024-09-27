
# Set variables
name="REFace"
device=2
CKPT="models/REFace/checkpoints/last.ckpt"

## CelebA ##
Results_dir="results/CelebA/${name}"
CONFIG="models/REFace/configs/project.yaml"


# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 3 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "CelebA" \
    --ddim_steps 50


## FFHQ ##
CONFIG="models/REFace/configs/project_ffhq.yaml"  
Results_dir="results/FFHQ/${name}"

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 3 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "FFHQ" \
    --ddim_steps 50


    



