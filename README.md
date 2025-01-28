# REFace

This repository gives the official implementation of Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models (WACV 2025 - Oral)

![Example](assets/teaser2.jpeg)
### [Paper](https://arxiv.org/abs/2409.07269)
[Sanoojan Baliah](https://www.linkedin.com/in/sanoojan/), Qinliang Lin, Shengcai Liao, Xiaodan Liang, and Muhammad Haris Khan

## Abstract
>Despite promising progress in face swapping task, realistic swapped images remain elusive, often marred by artifacts, particularly in scenarios involving high pose variation, color differences, and occlusion. To address these issues, we propose a novel approach that better harnesses diffusion models for face-swapping by making following core contributions. (a) We propose to re-frame the face-swapping task as a self-supervised, train-time inpainting problem, enhancing the identity transfer while blending with the target image. (b) We introduce a multi-step Denoising Diffusion Implicit Model (DDIM) sampling during training, reinforcing identity and perceptual similarities. (c) Third, we introduce CLIP feature disentanglement to extract pose, expression, and lighting information from the target image, improving fidelity. (d) Further, we introduce a mask shuffling technique during inpainting training, which allows us to create a so-called universal model for swapping, with an additional feature of head swapping. Ours can swap hair and even accessories, beyond traditional face swapping. Unlike prior works reliant on multiple off-the-shelf models, ours is a relatively unified approach and so it is resilient to errors in other off-the-shelf models. Extensive experiments on FFHQ and CelebA datasets validate the efficacy and robustness of our approach, showcasing high-fidelity, realistic face-swapping with minimal inference time. Our code is available here (https://github.com/Sanoojan/REFace)



## News
- *2024-09-10* Release training code
- *2024-09-10* Release test benchmark.
- *2024-09-14* Release checkpoints and other dependencies


## Requirements
A suitable [conda](https://conda.io/) environment named `REFace` can be created
and activated with:

```
conda create -n "REFace" python=3.10.13 -y
conda activate REFace
sh setup.sh
```


## Pretrained model

Download our trained model [here](https://huggingface.co/Sanoojan/REFace/blob/main/last.ckpt).

## Other dependencies 

Download the following models from the provided links and place them in the corresponding paths to perform face swapping and quantitative evaluation.



#### face parsing model (segmentation) 
[Other_dependencies/face_parsing/79999_iter.pth](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)

#### Arcface ID retrieval model 
[Other_dependencies/arcface/model_ir_se50.pth](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)

#### Landmark detection model 
[Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

#### Expression model (For quantitative analysis only) 
[Other_dependencies/face_recon/epoch_latest.pth](https://drive.google.com/file/d/1BlDBB4dLLrlN3cJhVL4nmrd_g6Jx6uP0/view?usp=drive_link)

[eval_tool/Deep3DFaceRecon_pytorch_edit/BFM/*.mat](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/master/BFM)


#### pose model (For quantitative analysis only)
[Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl](https://github.com/human-analysis/RankGAN/blob/master/models/hopenet_robust_alpha1.pkl)


### Alternatively, all the models can be downloaded directly from our [huggingface repo](https://huggingface.co/Sanoojan/REFace/tree/main) and replace the Other_dependencies folder, and eval_tool/Deep3DFaceRecon_pytorch_edit/BFM folder.


## Demo

To try our face-swapping inference on individual images using a graphical user interface, follow these steps:

```
sh Demo.sh
```
After launching, a link will be generated in the terminal. Open this link in your browser to access the GUI interface.
Use the interface to upload your source and target images. Simply select the images and click “Submit.”
Once processed, the output image will appear alongside the input images in the GUI.
Enjoy testing the Realistic and Efficient Face-Swapping (REFace) demo!


## Testing

To test our model on a dataset with facial masks (Follow dataset preparation), you can use `scripts/inference_test_bench.py`. For example, 
```
CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 3.5 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "CelebA" \
    --ddim_steps 50
```
or simply run:
```
sh inference_test_bench.sh
```
For a choosen folder of source and targets do faceswapping run this:
```
sh inference_selected.sh
```


## Training

### Data preparing
- Download CelebAHQ dataset

The data structure is like this:
```
dataset/FaceData
├── CelebAMask-HQ
│  ├── CelebA-HQ-img
│  │  ├── 0.png
│  │  ├── 1.png
│  │  ├── ...
│  ├── CelebA-HQ-mask
│  │  ├── Overall_mask
│  │  │   ├── 0.png
│  │  │   ├── ...
```

To create Overall mask from CelebAHQ datasets masks, simply use process_CelebA_mask.py

### Download the pretrained model of Stable Diffusion
We utilize the pretrained Stable Diffusion v1-4 as initialization, please download the pretrained models from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and save the model to directory `pretrained_models`. Then run the following script to add zero-initialized weights for 5 additional input channels of the UNet (4 for the encoded masked-image and 1 for the mask itself).
```
python scripts/modify_checkpoints.py
```

### Training REFace
To train a new model on CelebAHQ, you can use `main_swap.py`. For example,
```
python -u main_swap.py \
--logdir models/REFace/ \
--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
--base configs/train.yaml \
--scale_lr False 
```
or simply run:
```
sh train.sh
```

## Test Benchmark
We build a test benchmark for quantitative analysis. 

To prepare FFHQ masks, run

```
python esitmate_FFHQ_mask.py --seg12
```

To get the face swapping outcomes on CelebA and FFHQ datasets run,

```
bash inference_test_bench.sh
```


## Quantitative Results


By default we assume the original dataset images, selected source, target and masks and corresponding swapped images are generated. To evaluate the face swapping in terms if FID, ID retrieval, Pose and Expression simply run:

```
bash evaluate_all.sh
```


## Citing Us
If you find our work valuable, we kindly ask you to consider citing our paper and starring ⭐ our repository. Our implementation includes a standard metric code and we hope it make life easier for the generation research community.


```
@article{baliah2024realistic,
  title={Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models},
  author={Baliah, Sanoojan and Lin, Qinliang and Liao, Shengcai and Liang, Xiaodan and Khan, Muhammad Haris},
  journal={arXiv preprint arXiv:2409.07269},
  year={2024}
}
```

## Acknowledgements

This code borrows heavily from [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example).

## Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us. 

## License


This project is licensed under the MIT License. See LICENSE.txt for the full MIT license text.

Additional Notes:

Note 1: This project includes a derivative of [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example) licensed under the CreativeML Open RAIL-M license. The original license terms and use-based restrictions of the CreativeML Open RAIL-M license still apply to the model and its derivatives. Please refer to https://github.com/Fantasy-Studio/Paint-by-Example?tab=License-1-ov-file for more details.

Note 2: This work includes a model that has been trained using the CelebAMask-HQ dataset. The CelebAMask-HQ dataset is available for non-commercial research purposes only. As a result, any use of this model must comply with the non-commercial usage restriction of the CelebAMask-HQ dataset. Use of this model for commercial purposes is strictly prohibited.


