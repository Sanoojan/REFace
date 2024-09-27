"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from src.Face_models.encoders.model_irse import Backbone
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss, l1_loss
import torch.nn.utils as utils
import face_alignment
import torch

from eval_tool.Deep3DFaceRecon_pytorch.options.test_options import TestOptions
from pretrained.face_parsing.model import BiSeNet, seg_mean, seg_std
# from utils.module import SpecificNorm, cosin_metric


class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(SpecificNorm, self).__init__()
        # self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = np.array([0.5, 0.5, 0.5])
        self.mean = torch.from_numpy(self.mean).float().cuda()
        self.mean = self.mean.view([1, 3, 1, 1])

        # self.std = np.array([0.229, 0.224, 0.225])
        self.std = np.array([0.5, 0.5, 0.5])
        self.std = torch.from_numpy(self.std).float().cuda()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        x = (x - mean) / std
        return x

# give empty string to use the default options
dmm_defaults = TestOptions('')

dmm_defaults=dmm_defaults.parse()

from eval_tool.Deep3DFaceRecon_pytorch.models import create_model

def get_eye_coords(fa, image):
    image = image.squeeze(0)
    image = image * 128 + 128
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()

    try:
        preds = fa.get_landmarks(image)[0]
    except:
        return [None] * 8

    x, y = 5, 9
    left_eye_left = preds[36]
    left_eye_right = preds[39]
    eye_y_average = (left_eye_left[1] + left_eye_right[1]) // 2
    left_eye = [int(left_eye_left[0]) - x, int(eye_y_average - y), int(left_eye_right[0]) + x, int(eye_y_average + y)]
    right_eye_left = preds[42]
    right_eye_right = preds[45]
    eye_y_average = (right_eye_left[1] + right_eye_right[1]) // 2
    right_eye = [int(right_eye_left[0]) - x, int(eye_y_average - y), int(right_eye_right[0]) + x, int(eye_y_average + y)]
    return [*left_eye, *right_eye]

def get_eye_coords_from_landmarks(fa, landmarks):
    preds=landmarks if  landmarks[0][0] !=0 else None
    if preds is None:
        return [None] * 8
     
    x, y = 5, 9
    left_eye_left = preds[36]
    left_eye_right = preds[39]
    eye_y_average = (left_eye_left[1] + left_eye_right[1]) // 2
    left_eye = [int(left_eye_left[0]) - x, int(eye_y_average - y), int(left_eye_right[0]) + x, int(eye_y_average + y)]
    right_eye_left = preds[42]
    right_eye_right = preds[45]
    eye_y_average = (right_eye_left[1] + right_eye_right[1]) // 2
    right_eye = [int(right_eye_left[0]) - x, int(eye_y_average - y), int(right_eye_right[0]) + x, int(eye_y_average + y)]
    return [*left_eye, *right_eye]

def get_full_coords(fa,image):
    image = image
    # image = image * 128 + 128
    # image = image.to(torch.uint8)
    # image = image.permute(1, 2, 0)
    # image = image.cpu().numpy()

    try:
        preds = fa.face_alignment_net(image)
        # breakpoint()
    except:
        return [None] * 62
    return preds

from Other_dependencies.gaze_estimation.gaze_estimator import Gaze_estimator

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

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

class IDLoss(nn.Module):
    def __init__(self,path="Other_dependencies/arcface/model_ir_se50.pth",multiscale=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan
        
        self.facenet.load_state_dict(torch.load(path))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )
        
        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

    

    def forward(self, y_hat, y,clip_img=True,return_seperate=False):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y,clip_img=clip_img)  # Otherwise use the feature from there

        y_hat_feats_ms = self.extract_feats(y_hat,clip_img=clip_img)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]
        
        loss_all = 0
        sim_improvement_all = 0
        seperate_losses=[]
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
 
            loss = 0
            sim_improvement = 0
            count = 0
            
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])

                seperate_losses.append(1-sim_target)
                loss += 1 - sim_target  # id loss
                sim_improvement +=  float(sim_target) - float(sim_views)
                count += 1
            
            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
    
        return loss_all, sim_improvement_all, None
    

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.netGaze = Gaze_estimator().to(self.model.device)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        
        self.models_3dmm = create_model(dmm_defaults)
        self.models_3dmm.setup(dmm_defaults)
        
        if torch.cuda.is_available():
            self.models_3dmm.net_recon.cuda()
        
        self.seg = BiSeNet(n_classes=19)
        if torch.cuda.is_available():
            self.seg.to(self.model.device)

        self.seg.load_state_dict(torch.load("pretrained/face_parsing/79999_iter.pth"))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()
        
        self.spNorm = SpecificNorm()
        
        # self.ID_LOSS=IDLoss()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,src_im=None,tar=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    src_im=src_im,tar=tar,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,src_im=None,tar=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        if src_im is not None:
            src_im=un_norm_clip(src_im)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            outs = self.p_sample_ddim_guided_forward(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,src_im=src_im,tar=tar,**kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim_guided_forward(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,src_im=None,tar=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        
        torch.set_grad_enabled(True)
        x_in = x.detach().requires_grad_(True)
        
        
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x_in, t, c)
        else:  # check @ sanoojan
            x_in_n = torch.cat([x_in] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            e_t_uncond, e_t = self.model.apply_model(x_in_n, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x_in, t, c, **corrector_kwargs)

        


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x_in[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # G_id=ID_LOSS
            seperate_sim=None
            # src_im=None
            if src_im is not None:
                pred_x0_im=self.model.differentiable_decode_first_stage(pred_x0)
                masks=1-TF.resize(x_in[:,8,:,:],(pred_x0_im.shape[2],pred_x0_im.shape[3]))
                #mask x_samples_ddim
                pred_x0_im_masked=pred_x0_im*masks.unsqueeze(1)
                # x_samples_ddim_masked=un_norm_clip(x_samples_ddim_masked)
                # x_samples_ddim_masked = TF.normalize(x_samples_ddim_masked, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                Loss=0
                # breakpoint()
                # im_rec=torch.clamp((pred_x0_im_masked + 1.0) / 2.0, min=0.0, max=1.0)
                # im_tar=torch.clamp((tar + 1.0) / 2.0, min=0.0, max=1.0)
                # im_src=torch.clamp(src_im, min=0.0, max=1.0)
                
                
                # Segmentation Loss
                ################################
                # src_mask  = (pred_x0_im + 1) / 2
                # src_mask  = TF.resize(src_mask,(512,512))
                # src_mask  = TF.normalize(src_mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # targ_mask = (tar + 1) / 2
                # targ_mask  = TF.resize(targ_mask,(512,512))
                # targ_mask = TF.normalize(targ_mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # # breakpoint()
                # src_seg  = self.seg(self.spNorm(src_mask))[0]
                # # breakpoint()
                # src_seg = TF.resize(src_seg, (256, 256))
                # targ_seg = self.seg(self.spNorm(targ_mask))[0]
                # targ_seg = TF.resize(targ_seg, (256, 256))

                # seg_loss = torch.tensor(0).to(self.model.device).float()

                # # Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
                # ids = [1,11, 12, 13]

                # for id in ids:
                #     seg_loss += l1_loss(src_seg[:,id,:,:], targ_seg[:,id,:,:])
                #     # seg_loss += mse_loss(src_seg[0,id,:,:], targ_seg[0,id,:,:])

                # Loss = Loss + seg_loss * 10
                ################################
                
                #3DMM Loss
                ################################
                # im_rec=(pred_x0_im_masked + 1.0) / 2.0
                
                # #resize im_tar to 512x512
                
                # im_tar=(tar + 1.0) / 2.0
                # im_src=TF.resize(src_im,(512,512))
                
                # im_conc=torch.cat((im_rec,im_tar,im_src),dim=0)
                # c_all=self.models_3dmm.net_recon(im_conc)
                # c_rec=c_all[:b]
                # c_tar=c_all[b:2*b]
                # c_src=c_all[2*b:]
                
                
                # # breakpoint()
                # # c_rec=self.models_3dmm.net_recon(im_rec)
                # # c_tar=self.models_3dmm.net_recon(im_tar)
                # # c_src=self.models_3dmm.net_recon(im_src)
                
                # target_filt_c_rec=c_rec[:, 80:144]
                # target_filt_c_tar=c_tar[:, 80:144]
                
                # #select 0:80 and 144:224
                # src_filt_c_rec=torch.cat((c_rec[:, :80],c_rec[:, 144:224]),dim=1)
                # src_filt_c_src=torch.cat((c_src[:, :80],c_src[:, 144:224]),dim=1)
                
                
                # # src_filt_c_rec=c_rec[:, 144:224]
                # # src_filt_c_src=c_src[:, 144:224]
                
                
                # # id_coeffs = coeffs[:, :80]
                # # exp_coeffs = coeffs[:, 80: 144]
                # # tex_coeffs = coeffs[:, 144: 224]
                # # angles = coeffs[:, 224: 227]
                # # gammas = coeffs[:, 227: 254]
                # # translations = coeffs[:, 254:]
                        
                # # cosine loss between the two reconstructions
                # # cosine_loss =1- torch.nn.functional.cosine_similarity(target_filt_c_rec, target_filt_c_tar, dim=1)
                
                # # cosine_loss =1- torch.nn.functional.cosine_similarity(src_filt_c_rec, src_filt_c_src, dim=1)
                # cosine_loss_tar=1- torch.nn.functional.cosine_similarity(target_filt_c_rec, target_filt_c_tar, dim=1)
                # # L2_loss = mse_loss(src_filt_c_rec, src_filt_c_src)
                
                # Loss+=cosine_loss_tar.sum()*100.0
                # # +cosine_loss_tar.sum()*100.0
                ################################
                
                # GAZE Loss
                ################################
                # if t[0]>10 and t[0]<200:
                    
                #     src_eyes = pred_x0_im_masked * 0.5 + 0.5
                #     targ_eyes = tar
                #     targ_eyes = targ_eyes * 0.5 + 0.5
                #     for targ_eye,src_eye in zip(targ_eyes, src_eyes):
                #         targ_eye = targ_eye.unsqueeze(0)
                #         src_eye = src_eye.unsqueeze(0)
                #         try:
                #             llx, lly, lrx, lry, rlx, rly, rrx, rry = get_eye_coords(self.fa, targ_eye)
    
                #             if llx is not None:
                #                 targ_left_eye   = targ_eye[:, :, lly:lry, llx:lrx]
                #                 src_left_eye    = src_eye[:, :, lly:lry, llx:lrx]
                #                 targ_right_eye  = targ_eye[:, :, rly:rry, rlx:rrx]
                #                 src_right_eye   = src_eye[:, :, rly:rry, rlx:rrx]
                #                 targ_left_eye   = torch.mean(targ_left_eye, dim=1, keepdim=True)
                #                 src_left_eye    = torch.mean(src_left_eye, dim=1, keepdim=True)
                #                 targ_right_eye  = torch.mean(targ_right_eye, dim=1, keepdim=True)
                #                 src_right_eye   = torch.mean(src_right_eye, dim=1, keepdim=True)
                #                 targ_left_gaze  = self.netGaze(targ_left_eye.squeeze(0))
                #                 src_left_gaze   = self.netGaze(src_left_eye.squeeze(0))
                #                 targ_right_gaze = self.netGaze(targ_right_eye.squeeze(0))
                #                 src_right_gaze  = self.netGaze(src_right_eye.squeeze(0))
                #                 left_gaze_loss  = l1_loss(targ_left_gaze, src_left_gaze)
                #                 right_gaze_loss = l1_loss(targ_right_gaze, src_right_gaze)
                #                 gaze_loss = (left_gaze_loss + right_gaze_loss) * 1

                #                 Loss+=gaze_loss.sum()
                #         except:
                #             print("Error in Gaze Estimation")
                
                #ID LOSS        
                #####################
                # if t[0]>5 and t[0]<500:
                #     ID_loss,_,seperate_sim=self.model.face_ID_model(pred_x0_im_masked,src_im,clip_img=False,return_seperate=True)
                #     Loss+=ID_loss
                        # breakpoint()
                ######################################
                        
                # #Landmark_loss  Some problem with detected landmarks
                # if t[0]>5 and t[0]<300:
                #     src_eyes = pred_x0_im * 0.5 + 0.5
                #     targ_eyes = tar
                #     targ_eyes = targ_eyes * 0.5 + 0.5
                #     for targ_eye,src_eye in zip(targ_eyes, src_eyes):
                #         targ_eye = targ_eye.unsqueeze(0)
                #         src_eye = src_eye.unsqueeze(0)
                        
                #         try:
                #             preds = get_full_coords(self.fa, targ_eye)
                #             pred_swap = get_full_coords(self.fa, src_eye)

                #             # Extract batch size
                #             B = preds.shape[0]

                #             # Reshape to (B, 68, 128*128)
                #             preds_flat = preds.view(B, 68, -1)
                #             pred_swap_flat = pred_swap.view(B, 68, -1)

                #             # Find argmax indices
                #             preds_flat = torch.softmax(preds_flat, dim=-1)
                #             pred_swap_flat = torch.softmax(pred_swap_flat, dim=-1)

                #             breakpoint()
                #             # Calculate x and y indices
                #             # h, w = preds.shape[2], preds.shape[3]
                #             # indices_x = argmax_indices // w
                #             # indices_y = argmax_indices % w
                #             # indices_x_swap = argmax_indices_swap // w
                #             # indices_y_swap = argmax_indices_swap % w

                #             # Stack x and y indices
                #             # pred_coord = torch.stack((indices_x, indices_y), dim=-1).float()
                #             # pred_swap_coord = torch.stack((indices_x_swap, indices_y_swap), dim=-1).float()
                #             # breakpoint()
                #             # Select landmarks 48:67 and calculate MSE loss
                #             landmark_loss = mse_loss(preds_flat[:, 48:68, :], pred_swap_flat[:, 48:68, :])

                #             # Add the loss to the total loss
                #             Loss += landmark_loss 
                            
                #         except:
                #             print("Error in Landmark Estimation")
                ################################  
                
                
                
                
                
                # breakpoint()
                if Loss!=0:
                    grad=torch.autograd.grad(-1*Loss, x_in)[0]
                    grad=grad*5.0
                    grad=grad[:,:4,:,:].detach()
                    e_t=e_t-sqrt_one_minus_at*grad
                    x_in=x_in.requires_grad_(False)
                    del pred_x0,pred_x0_im_masked,grad,x_in,masks,pred_x0_im
                    torch.set_grad_enabled(False)
                else:
                    del pred_x0,pred_x0_im_masked,pred_x0_im
                    torch.set_grad_enabled(False)
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        with torch.no_grad():
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            del dir_xt,noise,x
            #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            # else:  
            #     seperate_sim=3*torch.tensor(seperate_sim)
            #     #make upper limit 1 and lower limit 0
            #     seperate_sim=torch.clamp(seperate_sim,0,1)
            #     x_prev = a_prev.sqrt() * pred_x0 + seperate_sim.view(-1,1,1,1).to(self.model.device)*dir_xt + noise
            return x_prev, pred_x0
        
    @torch.no_grad()
    def p_sample_ddim_guided(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,src_im=None,tar=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        
        torch.set_grad_enabled(True)
        x_in = x.detach().requires_grad_(True)
        
        
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x_in, t, c)
        else:  # check @ sanoojan
            x_in_n = torch.cat([x_in] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            e_t_uncond, e_t = self.model.apply_model(x_in_n, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x_in, t, c, **corrector_kwargs)

        


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        
        Backward_guidance=True
        # current prediction for x_0
        
        if Backward_guidance:
            pred_x0 = (x_in[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            recons_image=self.model.differentiable_decode_first_stage(pred_x0)
            recons_image=recons_image.detach().requires_grad_(True)
            masks=1-TF.resize(x_in[:,8,:,:],(recons_image.shape[2],recons_image.shape[3]))
            optimizer = torch.optim.Adam([recons_image], lr=5e-13)
            
            
            weights = torch.ones_like(recons_image).cuda()
            ones = torch.ones_like(recons_image).cuda()
            zeros = torch.zeros_like(recons_image).cuda()
            max_iters=2
            for _ in range(max_iters):
                with torch.no_grad():
                    recons_image.clamp_(-1, 1)

                optimizer.zero_grad()
                # if operation_func != None:
                #     op_im = operation_func(recons_image)
                # else:
                #     op_im = recons_image

                # loss = criterion(op_im, operated_image)
                src_im=TF.normalize(src_im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ID_loss,_,seperate_sim=self.model.face_ID_model(recons_image,src_im,clip_img=False,return_seperate=True)
                loss=1-torch.stack(seperate_sim,dim=0)
                
                
                
                src_mask  = (recons_image + 1) / 2
                src_mask  = TF.resize(src_mask,(512,512))
                src_mask  = TF.normalize(src_mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                targ_mask = (tar + 1) / 2
                targ_mask  = TF.resize(targ_mask,(512,512))
                targ_mask = TF.normalize(targ_mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # breakpoint()
                src_seg  = self.seg(self.spNorm(src_mask))[0]
                # breakpoint()
                src_seg = TF.resize(src_seg, (256, 256))
                targ_seg = self.seg(self.spNorm(targ_mask))[0]
                targ_seg = TF.resize(targ_seg, (256, 256))

                # seg_loss = torch.tensor(0).to(self.model.device).float()

                # Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
                ids = [11, 12, 13]
                seg_losses=[]
                for im in range(recons_image.shape[0]):
                    seg_loss = 0
                    for id in ids:
                        seg_loss+=(l1_loss(src_seg[im,id,:,:], targ_seg[im,id,:,:]))
                    seg_losses.append(seg_loss)
                loss+=torch.stack(seg_losses,dim=0)
                # for id in ids:
                #     seg_loss += l1_loss(src_seg[:,id,:,:], targ_seg[:,id,:,:])
                    # seg_loss += mse_loss(src_seg[0,id,:,:], targ_seg[0,id,:,:])

                # seg_loss * 200
                
                
                
                # breakpoint()
                for __ in range(loss.shape[0]):
                    if loss[__] < 0.00001:              #loss cutoff
                        weights[__] = zeros[__]
                    else:
                        weights[__] = ones[__]

                before_x = torch.clone(recons_image.data)


                m_loss = loss.mean()
                m_loss.backward()
                
                # breakpoint()
                utils.clip_grad_norm_(recons_image, 0.01)
                optimizer.step()

                # if operation.lr_scheduler != None:
                #     scheduler.step()
                # breakpoint()
                with torch.no_grad():
                    recons_image.data = before_x * (1 - weights) + weights * recons_image.data

                if weights.sum() == 0:
                    break
                
            recons_image.requires_grad = False
            torch.set_grad_enabled(False)

            recons_image = torch.clamp(recons_image, -1, 1)
            pred_x0_new = self.model.encode_first_stage(recons_image)
            
            pred_x0_new=self.model.get_first_stage_encoding(pred_x0_new)
            # breakpoint()
            e_t=(x[:,:4,:,:]- pred_x0_new*a_t.sqrt())/sqrt_one_minus_at
            
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        with torch.no_grad():
            if Backward_guidance:
                pred_x0=pred_x0_new
            else:
                pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            
            
            del dir_xt,noise,x
            #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            # else:  
            #     seperate_sim=3*torch.tensor(seperate_sim)
            #     #make upper limit 1 and lower limit 0
            #     seperate_sim=torch.clamp(seperate_sim,0,1)
            #     x_prev = a_prev.sqrt() * pred_x0 + seperate_sim.view(-1,1,1,1).to(self.model.device)*dir_xt + noise
            return x_prev, pred_x0
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:  # check @ sanoojan
            x_in = torch.cat([x] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    
    def sample_train(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               t=None,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling_train(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,ddim_num_steps=S,
                                                    curr_t=t,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

 
    def ddim_sampling_train(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,ddim_num_steps=None,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,curr_t=None,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
            
        kwargs['rest']=img[:,4:,:,:]
        img=img[:,:4,:,:]
        

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        curr_t=curr_t.cpu().numpy()
        skip = (curr_t-1) // ddim_num_steps
        # replace all 0s with 1s
        skip[skip == 0] = 1
        if type(skip)!=int:
            seq=[range(1, curr_t[n]-1, skip[n]) for n in range(len(curr_t))]
            min_length = min(len(sublist) for sublist in seq)
            min_length=min(min_length,ddim_num_steps)
            # Create a new list of sublists by truncating each sublist to the minimum length
            truncated_seq = [sublist[:min_length] for sublist in seq]
            seq= np.array(truncated_seq)

            # seq=np.flip(seq)
        #concatenate all sequences
        # seq = np.concatenate(seq)
        seq=torch.from_numpy(seq).to(device)
        seq=torch.flip(seq,dims=[1])

        
        
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")


        # time_range=np.array([1])
        # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        total_steps=seq.shape[1]
        for i in range(seq.shape[1]):
            index = total_steps - i - 1
            # ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts=seq[:,i].long()
            #make it toech long
            # ts=ts.long()

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            outs = self.p_sample_ddim_train(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,**kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim_train(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,return_features=False,**kwargs):
        b, *_, device = *x.shape, x.device
        # if 'test_model_kwargs' in kwargs:
        #     kwargs=kwargs['test_model_kwargs']
        #     x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        if 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
    
            
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c,return_features=return_features)
        else:  # check @ sanoojan
            x_in = torch.cat([x] * 2) #x_in: 2,9,64,64
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c]) #c_in: 2,1,768
            if return_features:
                e_t_uncond, e_t,features = self.model.apply_model(x_in, t_in, c_in,return_features=return_features).chunk(3)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) #1,4,64,64

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim_guided(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec