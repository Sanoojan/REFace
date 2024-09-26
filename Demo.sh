
# Set variables
name="One_output"
Results_dir="examples/FaceSwap/${name}/results"
Base_dir="examples/FaceSwap/${name}/Outs"
Results_out="examples/FaceSwap/${name}/results/results" 
device=0

CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/checkpoints/saved.ckpt"

#change this
target_path="examples/FaceSwap/One_target"  
source_path="examples/FaceSwap/One_source"


# Run inference
# ideal for small number of samples

CUDA_VISIBLE_DEVICES=${device} python scripts/one_inference.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 1 \
    --scale 3.5 \
    --ddim_steps 50



