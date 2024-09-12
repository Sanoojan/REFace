
# Set variables
name="v5_Two_CLIP_proj_154_ep_3.5_5"
Results_dir="results_FINALS/${name}"
Results_out="results_FINALS/${name}/results"
Write_results="Quantitative_FINALS/P4s/${name}"
device=3

CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"
source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"
target_mask_path="dataset/FaceData/CelebAMask-HQ/target_mask"
Dataset_path="dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi

# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 3.5 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "CelebA" \
    --ddim_steps 50

    


echo "FID score with Dataset:" >> "$output_filename"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    "${Dataset_path}" \
    "${Results_out}"  >> "$output_filename"

echo "Pose comarison with target:" >> "$output_filename"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
    "${target_path}" \
    "${Results_out}"  >> "$output_filename"

echo "Expression comarison with target:" >> "$output_filename"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
    "${target_path}" \
    "${Results_out}"  >> "$output_filename"


echo "ID similarity with Source using Arcface:" >> "$output_filename"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}" \
    --print_sim True  \
    --arcface True >> "$output_filename"   


now_time=$(date +"%Y%m%d_%H%M%S")
elapsed_time=$(($(date -d $now_time +%s) - $(date -d $current_time +%s)))
echo "Elapsed time in minutes: $((elapsed_time / 60))" >> "$output_filename"
