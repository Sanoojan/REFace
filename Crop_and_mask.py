from src.utils.alignmengt import crop_faces, calc_alignment_coefficients, crop_faces_from_image
from PIL import Image
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
from tqdm import tqdm
def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def crop_and_align_face_img(frame):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces_from_image(image_size, frame, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results_video/debug"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
        default="True"
    )
    parser.add_argument(
        "--Start_from_target",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
    )
    parser.add_argument(
        "--only_target_crop",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
        default=True
    )
    parser.add_argument(
        "--target_start_noise_t",
        type=int,
        default=1000,
        help="target_start_noise_t",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--target_video",
        type=str,
        help="target_video",
        default="examples/faceswap/Andy2.mp4",
    )
    parser.add_argument(
        "--src_image",
        type=str,
        help="src_image",
        default="examples/faceswap/source.jpg"
    )
    parser.add_argument(
        "--src_image_mask",
        type=str,
        help="src_image_mask",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/REFace/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
    parser.add_argument('--faceParsing_ckpt', type=str, default="Other_dependencies/face_parsing/79999_iter.pth")  
    parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                        'this option is valid when --faceParsing_ckpt=segenext')
            
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--seg12',default=True, action='store_true')
    
    opt = parser.parse_args()


    faceParsing_model = init_faceParsing_pretrained_model(opt.faceParser_name, opt.faceParsing_ckpt, opt.segnext_config)
        

    Image_path='dataset/FaceData/CelebAMask-HQ/Val'
    mask_real_path='dataset/FaceData/CelebAMask-HQ/src_mask'
    mask_path='dataset/FaceData/CelebAMask-HQ/Val_cropped_mask'
    save_path='dataset/FaceData/CelebAMask-HQ/Val_cropped'
    
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get image list
    image_list = glob.glob(os.path.join(Image_path, '*.png'))

    for i in tqdm(range(29000,30000)):
        try:
            image_path_name=Image_path+ "/" +str(i)+ '.jpg'
            save_path_name=save_path+ "/" +str(i)+ '.jpg'
            mask_path_name=mask_path+ "/" +str(i)+ '.png'
            mask_real_path_name=mask_real_path+ "/" +str(i)+ '.png'
            frame = cv2.imread(image_path_name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Image.fromarray(frame).save(os.path.join(Image_path, f'{i}.png'))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crops, orig_images, quads, inv_transforms = crop_and_align_face([image_path_name])
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
            # inv_transforms_all.append(inv_transforms[0])
            T.save(save_path_name)
            
            real_mask = cv2.imread(mask_real_path_name)
            if real_mask is None:
                pil_im = T.resize((1024,1024), Image.BILINEAR)
                mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
                
                Image.fromarray(mask).save(mask_path_name)
                # save T
                
            else:
                
                real_mask = Image.fromarray(real_mask)
                real_mask=real_mask.resize((1024,1024), Image.BILINEAR)
                #crop using quads
                real_mask = real_mask.crop((quads[0][0][0], quads[0][0][1], quads[0][2][0], quads[0][2][1]))
                
                real_mask=real_mask.resize((512,512), Image.BILINEAR)
                real_mask.save(mask_path_name)
                # save T
           
        except:
            print("error")
            # read image as pil
            T = Image.open(image_path_name)
            # inv_transforms_all.append(inv_transforms[0])
            real_mask = cv2.imread(mask_real_path_name)
            T.save(save_path_name)
            if real_mask is None:
                pil_im = T.resize((1024,1024), Image.BILINEAR)
                mask = faceParsing_demo(faceParsing_model, pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
                
                Image.fromarray(mask).save(mask_path_name)
                # save T
                
            else:
                
                real_mask = Image.fromarray(real_mask)
                real_mask=real_mask.resize((1024,1024), Image.BILINEAR)
                #crop using quads
                real_mask = real_mask.crop((quads[0][0][0], quads[0][0][1], quads[0][2][0], quads[0][2][1]))
                
                real_mask=real_mask.resize((512,512), Image.BILINEAR)
                real_mask.save(mask_path_name)
                
            # continue
if __name__ == "__main__":
    main()