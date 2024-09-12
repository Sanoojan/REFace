
# Set variables
name="Swap_outs"
Results_dir="examples/FaceSwap/${name}/results"
Base_dir="examples/FaceSwap/${name}/Outs"
Results_out="examples/FaceSwap/${name}/results/results" 
device=1


CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"

target_path="examples/FaceSwap/Target"
source_path="examples/FaceSwap/Source"




# Run inference
# ideal for small number of samples

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_selected.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 4 \
    --scale 3.5 \
    --ddim_steps 50



