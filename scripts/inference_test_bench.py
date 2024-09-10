import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
# import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_guided import DDIMSampler as DDIMSampler_guided
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ldm.data.test_bench_dataset import COCOImageDataset
from ldm.data.test_bench_dataset import CelebAdataset,FFHQdataset,FFdataset
# import clip
from torchvision.transforms import Resize


from PIL import Image
from torchvision.transforms import PILToTensor




# from dift.src.models.dift_sd import SDFeaturizer
# from dift.src.utils.visualization import Demo


# import matplotlib.pyplot as plt
import torch.nn as nn

# cos = nn.CosineSimilarity(dim=0)
import numpy as np  

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# set cuda device 

def save_sample_by_decode(x,model,Base_path,segment_id_batch,intermediate_num):

    
    x = model.decode_first_stage(x)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    for i in range(len(x)):
        img = Image.fromarray((x[i] * 255).astype(np.uint8))
        save_path=os.path.join(Base_path, f"{segment_id_batch[i]}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img.save(os.path.join(Base_path, f"{segment_id_batch[i]}/{intermediate_num}.png"))
    


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


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
        "--device_ID",
        type=int,
        default=5,
        help="device_ID",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results/debug"
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
    )
    parser.add_argument(
        "--Guidance",
        action='store_true',
        help="Guidance in inference ",
    )
    parser.add_argument(
        "--Start_from_target",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
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
        default=5,
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
        "--dataset",
        type=str,
        help="dataset: CelebA,FFHQ,FF",
        default='CelebA'
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="dataset_dir",
        default='dataset/FaceData/CelebAMask-HQ'
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/Paint-by-Example/v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS/PBE/celebA/2024-02-05T22-04-36_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/configs/2024-02-05T22-04-36-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/Paint-by-Example/v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS/PBE/celebA/2024-02-05T22-04-36_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/checkpoints/last.ckpt",
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
        default="full"
    )
    opt = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print(opt)
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    # torch.cuda.set_device(opt.device_ID)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        if opt.Guidance:
            sampler = DDIMSampler_guided(model)
        else:
            sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

 

    # test_dataset=COCOImageDataset(test_bench_dir='test_bench') 
    #read config file :configs/v2.yaml
    conf_file=OmegaConf.load(opt.config)
    # breakpoint()
    test_args=conf_file.data.params.test.params
    
    if opt.dataset=='CelebA':
        test_dataset=CelebAdataset(split='test',**test_args)
        
    elif opt.dataset=='FFHQ':
        test_dataset=FFHQdataset(split='test',**test_args)
        
    elif opt.dataset=='FF++':
        test_args['dataset_dir']=opt.dataset_dir if opt.dataset_dir is not None else test_args['dataset_dir']
        test_dataset=FFdataset(split='test',**test_args)
        
    test_dataloader= torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=4, 
                                        pin_memory=True, 
                                        shuffle=False,
                                        drop_last=False)





    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

   
    use_prior=True
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    sample=0
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for test_batch,prior, test_model_kwargs,segment_id_batch in test_dataloader:
                    sample+=opt.n_samples
                    # if sample<980:
                    #     continue
                    if opt.Start_from_target:
                        
                        x=test_batch
                        x=x.to(device)
                        encoder_posterior = model.encode_first_stage(x)
                        z = model.get_first_stage_encoding(encoder_posterior)
                        t=int(opt.target_start_noise_t)
                        # t = torch.ones((x.shape[0],), device=device).long()*t
                        t = torch.randint(t-1, t, (x.shape[0],), device=device).long()
                    
                        if use_prior:
                            prior=prior.to(device)
                            encoder_posterior_2=model.encode_first_stage(prior)
                            z2 = model.get_first_stage_encoding(encoder_posterior_2)
                            noise = torch.randn_like(z2)
                            x_noisy = model.q_sample(x_start=z2, t=t, noise=noise)
                            start_code = x_noisy
                            # print('start from target')
                        else:
                            noise = torch.randn_like(z)
                            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
                            start_code = x_noisy
                        # print('start from target')
                        
                    test_model_kwargs={n:test_model_kwargs[n].to(device,non_blocking=True) for n in test_model_kwargs }
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector.repeat(test_batch.shape[0],1,1)
                        if model.stack_feat:
                            uc2=model.other_learnable_vector.repeat(test_batch.shape[0],1,1)
                            uc=torch.cat([uc,uc2],dim=-1)
                    
                    # c = model.get_learned_conditioning(test_model_kwargs['ref_imgs'].squeeze(1).to(torch.float16))
                    landmarks=model.get_landmarks(test_batch) if model.Landmark_cond else None
                    c=model.conditioning_with_feat(test_model_kwargs['ref_imgs'].squeeze(1).to(torch.float32),landmarks=landmarks,tar=test_batch.to("cuda").to(torch.float32)).float()
                    if (model.land_mark_id_seperate_layers or model.sep_head_att) and opt.scale != 1.0:
            
                        # concat c, landmarks
                        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                        uc=torch.cat([uc,landmarks],dim=-1)
                    
                    
                    if c.shape[-1]==1024:
                        c = model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)
                    inpaint_image=test_model_kwargs['inpaint_image']
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    # breakpoint()
                    samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=test_batch.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        log_every_t=100,
                                                        test_model_kwargs=test_model_kwargs,src_im=test_model_kwargs['ref_imgs'].squeeze(1).to(torch.float32),tar=test_batch.to("cuda"))
                    # breakpoint()
                    save_intermediates=False
                    
                    
                    # breakpoint()
                    if save_intermediates:
                        intermediate_pred_x0=intermediates['pred_x0']
                        intermediate_noised=intermediates['x_inter']
                        for i in range(len(intermediate_pred_x0)):
                            save_sample_by_decode(intermediate_pred_x0[i],model,Base_path="Save_intermediates/pred_x0",segment_id_batch=segment_id_batch,intermediate_num=i)
                            save_sample_by_decode(intermediate_noised[i],model,Base_path="Save_intermediates/intermediate",segment_id_batch=segment_id_batch,intermediate_num=i)
                    
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x1):
                        x = x1*1.0 # to avoid changing the original tensor or clone() can be used
                        reduce=False
                        if len(x.shape)==3:
                            x = x.unsqueeze(0)
                            reduce=True
                        x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
                        x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
                        x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
                        
                        if reduce:
                            x = x.squeeze(0)
                        return x

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            

                            all_img=[]
                            all_img.append(un_norm(test_batch[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=test_model_kwargs['ref_imgs'].squeeze(1)
                            ref_img=Resize([512,512])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample)
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(grid_path, 'grid-'+segment_id_batch[i]+'.png'))
                            



                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(result_path, segment_id_batch[i]+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, segment_id_batch[i]+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, segment_id_batch[i]+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, segment_id_batch[i]+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, segment_id_batch[i]+"_ref.png"))
                            base_count += 1



                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

