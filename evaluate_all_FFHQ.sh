#!/bin/bash


res_end="results"
results_start=""
Write_results="Quantitative_Grad_Other_swappers/FFHQ"



declare -a names=(  
                    # "/home/sanoojan/other_swappers/MegaFs/FFHQ_outs"
                    # "/home/sanoojan/other_swappers/hififace/FFHQ_results"
                    /home/sanoojan/Paint_for_swap/results_FFHQ_FINAL/v5_Two_CLIP_proj_154_last_ep/results
                    # "/home/sanoojan/e4s/Results/testbench/results_on_FFHQ_orig_ckpt/results"                
                    # "/home/sanoojan/other_swappers/SimSwap/output/FFHQ/results"
                    # "/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ-FFHQ/results"
                    # "/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_FFHQ"
                    # "/home/sanoojan/other_swappers/DiffFace/results/FFHQ/results"
                    )



device=0

source_path="dataset/FaceData/FFHQ/Val"
target_path="dataset/FaceData/FFHQ/Val_target"
source_mask_path="dataset/FaceData/FFHQ/src_mask"
target_mask_path="dataset/FaceData/FFHQ/target_mask"
Dataset_path="dataset/FaceData/FFHQ/images512"

for name in "${names[@]}"
do
    # Results_out="${results_start}/${name}/${res_end}"
    # Results_out="${name}/${res_end}"
    Results_out="${name}"
    current_time=$(date +"%Y%m%d_%H%M%S")
    Write_results_n="${Write_results}/${name}"
    output_filename="${Write_results_n}/out_${current_time}.txt"
    
    if [ ! -d "$Write_results_n" ]; then
        mkdir -p "$Write_results_n"
        echo "Directory created: $Write_results_n"
    else
        echo "Directory already exists: $Write_results_n"
    fi


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
        "${Results_out}" 
        # >> "$output_filename"

    echo "ID similarity with Source using Arcface:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${source_path}" \
        "${Results_out}" \
        "${source_mask_path}" \
        "${target_mask_path}" \
        --dataset "ffhq" \
        --print_sim True  \
        --arcface True >> "$output_filename" 
done
