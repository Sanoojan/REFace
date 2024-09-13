# REFace

This repository gives the official implementation of Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models (WACV 2025)

![Example](assets/teaser2.jpeg)
### [Paper](https://arxiv.org/abs/2409.07269)
[Sanoojan Baliah](https://www.linkedin.com/in/sanoojan/), Qinliang Lin, Shengcai Liao, Xiodan Liang, and Muhammad Haris Khan

## Abstract
>Despite promising progress in face swapping task, realistic swapped images remain elusive, often marred by artifacts, particularly in scenarios involving high pose variation, color differences, and occlusion. To address these issues, we propose a novel approach that better harnesses diffusion models for face-swapping by making following core contributions. (a) We propose to re-frame the face-swapping task as a self-supervised, train-time inpainting problem, enhancing the identity transfer while blending with the target image. (b) We introduce a multi-step Denoising Diffusion Implicit Model (DDIM) sampling during training, reinforcing identity and perceptual similarities. (c) Third, we introduce CLIP feature disentanglement to extract pose, expression, and lighting information from the target image, improving fidelity. (d) Further, we introduce a mask shuffling technique during inpainting training, which allows us to create a so-called universal model for swapping, with an additional feature of head swapping. Ours can swap hair and even accessories, beyond traditional face swapping. Unlike prior works reliant on multiple off-the-shelf models, ours is a relatively unified approach and so it is resilient to errors in other off-the-shelf models. Extensive experiments on FFHQ and CelebA datasets validate the efficacy and robustness of our approach, showcasing high-fidelity, realistic face-swapping with minimal inference time. Our code is available here (https://github.com/Sanoojan/REFace)



## News
- *2024-09-10* Release training code
- *2023-09-10* Release test benchmark.

## Todo
- Upload checkpoints
- Upload other dependencies
- Clean code



## Requirements
A suitable [conda](https://conda.io/) environment named `REFace` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate REFace
```

## Pretrained Model

For inference purpose


## Testing

To test our model on a dataset with facial masks (), you can use `scripts/inference_test_bench.py`. For example, 
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

### Download the pretrained model of Stable Diffusion
We utilize the pretrained Stable Diffusion v1-4 as initialization, please download the pretrained models from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and save the model to directory `pretrained_models`. Then run the following script to add zero-initialized weights for 5 additional input channels of the UNet (4 for the encoded masked-image and 1 for the mask itself).
```
python scripts/modify_checkpoints.py
```

### Training Paint by Example
To train a new model on Open-Images, you can use `main.py`. For example,
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

## Quantitative Results
By default, we assume that the COCOEE is downloaded and saved to the directory `test_bench`. To generate the results of test bench, you can use `scripts/inference_test_bench.py`. For example, 

or simply run:
```
bash inference_test_bench.sh
```


## Citing Us

```
@misc{baliah2024realisticefficientfaceswapping,
title={Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models},
author={Sanoojan Baliah and Qinliang Lin and Shengcai Liao and Xiaodan Liang and Muhammad Haris Khan},
year={2024},
eprint={2409.07269},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2409.07269},
}
```

## Acknowledgements

This code borrows heavily from [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example).

## Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us.

## License

(MIT)See License 