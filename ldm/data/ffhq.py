from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Any, Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from einops import rearrange
from torchvision.utils import save_image



import warnings
warnings.filterwarnings("ignore")


from torchvision.transforms import ToTensor, ToPILImage
# sys.path.append('/data/pzb/EAT/attack')
from thinplatespline.batch import TPS
from thinplatespline.tps import tps_warp
TOTEN = ToTensor()
TOPIL = ToPILImage()
DEVICE = torch.device("cpu")

# def grid_points_2d(width, height):
#     """
#     Create 2d grid points. Distribute points across a width and height,
#     with the resulting coordinates constrained to -1, 1
#     returns tensor shape (width * height, 2)
#     """
#     xx, yy = torch.meshgrid(
#         [torch.linspace(-1.0, 1.0, height),
#          torch.linspace(-1.0, 1.0, width)])
#     return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)
# def noisy_grid(width, height, noise_matrix):
#     """
#     Make uniform grid points, and add noise except for edge points.
#     """
#     grid = grid_points_2d(width, height)
#     mod = torch.zeros([height, width, 2])
#     mod[1:height - 1, 1:width - 1, :] = noise_matrix
#     return grid + mod.reshape(-1, 2)
# def grid_to_img(grid_points, width, height):
#     """
#     convert (N * 2) tensor of grid points in -1, 1 to tuple of (x, y)
#     scaled to width, height.
#     return (x, y) to plot"""
#     grid_clone = grid_points.clone().numpy()
#     x = (1 + grid_clone[..., 0]) * (width - 1) / 2
#     y = (1 + grid_clone[..., 1]) * (height - 1) / 2
#     return x.flatten(), y.flatten()
# def decow(img,a=4):
#     n, c, w, h = img.size()
#     device = torch.device('cuda')
#     # a = 4
#     X = grid_points_2d(a, a,)
#     noise = (torch.rand([a-2, a-2, 2]) - 0.5) * 0.9
#     # noise = (torch.rand([1, 1, 2]) - 0.5)
#     Y = noisy_grid(a, a, noise, device)
#     tpsb = TPS(size=(h, w), device=device)
#     warped_grid_b = tpsb(X[None, ...], Y[None, ...])
#     warped_grid_b = warped_grid_b.repeat(img.shape[0], 1, 1, 1)
#     awt_img = torch.grid_sampler_2d(img, warped_grid_b, 0, 0, False)
#     return awt_img

def grid_points_2d(width, height, device=DEVICE):
    """
    Create 2d grid points. Distribute points across a width and height,
    with the resulting coordinates constrained to -1, 1
    returns tensor shape (width * height, 2)
    """
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
         torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)
def noisy_grid(width, height, noise_matrix, device=DEVICE):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_matrix
    return grid + mod.reshape(-1, 2)
def grid_to_img(grid_points, width, height):
    """
    convert (N * 2) tensor of grid points in -1, 1 to tuple of (x, y)
    scaled to width, height.
    return (x, y) to plot"""
    grid_clone = grid_points.clone().detach().cpu().numpy()
    x = (1 + grid_clone[..., 0]) * (width - 1) / 2
    y = (1 + grid_clone[..., 1]) * (height - 1) / 2
    return x.flatten(), y.flatten()
def decow(img,scale=0.8):
    n, c, w, h = img.size()
    device = torch.device('cpu')
    a = 3
    X = grid_points_2d(a, a, device)
    noise = (torch.rand([a-2, a-2, 2]) - 0.5) * scale
    # noise = (torch.rand([1, 1, 2]) - 0.5)
    Y = noisy_grid(a, a, noise, device)
    tpsb = TPS(size=(h, w), device=device)
    warped_grid_b = tpsb(X[None, ...], Y[None, ...])
    warped_grid_b = warped_grid_b.repeat(img.shape[0], 1, 1, 1)
    awt_img = torch.grid_sampler_2d(img, warped_grid_b, 0, 0, False)
    return awt_img


# image_path = 'dataset/FaceData/CelebAMask-HQ/CelebA-HQ-mask/2/05998_skin.png'
# image = Image.open(image_path)


# transform = transforms.Compose([
#     # transforms.Resize((224, 224)),  # 
#     transforms.ToTensor(),  # 
# ])

# tensor = transform(image)
# mask = tensor.unsqueeze(0)  #


# print(tensor.shape)


# mask = decow(mask.cuda(),a=4)
# tensor_to_image = transforms.ToPILImage()
# image = tensor_to_image(mask.cpu().squeeze())
#

# image.save('face-aug.png')
















def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


#####

# 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow, 8:l_ear, 9:r_ear, 
# 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 16:neck_l, 17:neck, 18:cloth

# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
face_parsing_PyTorch_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...
 
# 9 attributes with left-right aggrigation
faceParser_label_list = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 
                         'nose', 'skin', 'ears', 'belowface']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

#FFHQ/ faceparcing network
# 0:background, 1:lip, 2:eyebrows, 3:eyes, 4:hair, 5:nose, 6:skin, 7:ears, 
# 8:belowface, 9:mouth, 10:eye_glass, 11:ear_rings
# preserve = [1,2,3,5,6,7,8,9]

TO_TENSOR = transforms.ToTensor()
MASK_CONVERT_TF = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask(celebAHQ_mask))

MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask_detailed(celebAHQ_mask))


NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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

    
def un_norm(x):
    return (x+1.0)/2.0

def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def __celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (12-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """
    # 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
    celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                            'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                            'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                            'neck_l', 'neck', 'cloth']# 12 attributes with left-right aggrigation
    faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                    'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    # print('skin', np.sum(skin))
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)
    converted_mask[ear_rings] = 11
    
    return converted_mask

def __celebAHQ_masks_to_faceParser_mask(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (9-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """

    assert len(celebA_mask.size) == 2, "The provided mask should be with [H,W] format"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(celebA_mask, 10), np.equal(celebA_mask, 11)),
        np.equal(celebA_mask, 12)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8

    return converted_mask



class FFHQdataset(data.Dataset):

    def __init__(self,state,arbitrary_mask_percent=0,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3), 
            # A.GaussNoise(p=0.3),# newly added from this line
            # A.HueSaturationValue(p=0.3),
            # A.ISONoise(p=0.3),
            # A.Solarize(p=0.3),
            ])
        
        self.gray_outer_mask=args['gray_outer_mask']
        self.preserve=args['preserve_mask_src']
        
        
        # print("gray_outer_mask",self.gray_outer_mask)
        # print("preserve_mask",self.preserve)
        # print("..........................................")
        
        self.Fullmask=False
        
        self.bbox_path_list=[]
        if state == "train":
            self.imgs = sorted([osp.join(args['dataset_dir'], "images512", '{0:0=5d}.png'.format(idx)) for idx in range(68000)])
            self.labels = sorted([osp.join(args['dataset_dir'], "BiSeNet_mask",'{0:0=5d}.png'.format(idx))  for idx in range(68000)])

        elif state == "validation":
            self.imgs = sorted([osp.join(args['dataset_dir'], "images512", '{0:0=5d}.png'.format(idx)) for idx in range(68000,70000)])
            self.labels = sorted([osp.join(args['dataset_dir'], "BiSeNet_mask",'{0:0=5d}.png'.format(idx))  for idx in range(68000,70000)])
            
        else:
            self.imgs = sorted([osp.join(args['dataset_dir'], "images512",'{0:0=5d}.png'.format(idx))  for idx in range(68000,69000)])
            self.labels = sorted([osp.join(args['dataset_dir'], "BiSeNet_mask", '{0:0=5d}.png'.format(idx))  for idx in range(68000,69000)]) 
            

            self.ref_imgs = sorted([osp.join(args['dataset_dir'], "images512", '{0:0=5d}.png'.format(idx))  for idx in range(69000,70000)])
            self.ref_labels =  sorted([osp.join(args['dataset_dir'], "BiSeNet_mask", '{0:0=5d}.png'.format(idx))  for idx in range(69000,70000)]) 
            

            self.ref_imgs= self.ref_imgs[:int(len(self.imgs)*self.fraction)]
            self.ref_labels= self.ref_labels[:int(len(self.labels)*self.fraction)]

            # if self.load_prior:
            #     self.prior_images=sorted([osp.join("intermediate_results_FFHQ_261/results", "0000000%d.jpg"%idx) for idx in range(68000, 69000)])
            
        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)

    def __getitem__(self, index):
        if self.gray_outer_mask:
            return self.__getitem_gray__(index)
        else:
            return self.__getitem_black__(index)


    def __getitem_gray__(self, index):

        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
     

        ############
        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        
        if self.Fullmask:
            mask_img_full=mask_img
            mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already
        
        
            
        
        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ]
        preserve = self.preserve # full mask to be changed
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
 
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
    
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np
        # resize mask_img
       
    
        
        # ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor)
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
       

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        image_tensor_cropped=image_tensor
        mask_tensor_cropped=mask_tensor
        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        
        # a=random.randint(1,4)
        scale=random.uniform(0.5, 1.0)
        mask_tensor_resize=decow(mask_tensor_resize.unsqueeze(0) ,scale=scale).squeeze(0)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        mask_ref=1-T.Resize([512,512])(mask_tensor)
        ref_image_tensor=ref_image_tensor*mask_ref
        
        # ref_image_tensor=Image.fromarray(ref_image_tensor)
        ref_image_tensor=255.* rearrange(un_norm_clip(ref_image_tensor), 'c h w -> h w c').cpu().numpy()
        
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor['image'].astype(np.uint8)) 
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
   
        if self.Fullmask:
            return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor}
   
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __getitem_black__(self, index):
        # black mask
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
     

        ############
        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ]
        preserve = self.preserve # full mask to be changed
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
 
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
    
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np
        # resize mask_img
        mask_img_r = mask_img.resize(img_p_np.shape[1::-1], Image.NEAREST)
        mask_img_r = np.array(mask_img_r)
        
        # select only mask_img region from reference image
        ref_image_tensor[mask_img_r==0]=0   # comment this if full img should be used
    
        
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        image_tensor_cropped=image_tensor
        mask_tensor_cropped=mask_tensor
        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
   
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}
   
   
    def __getitem_old__(self, index):

        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
        # if self.img_transform is not None:
        #     img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        # Assuming that 'label' is your binary mask (black and white image)
        label = np.array(label)  # Convert the label to a NumPy array if it's not already

        # Find the coordinates of the non-zero (white) pixels in the mask
        non_zero_coords = np.column_stack(np.where(label == 1))

        # Find the minimum and maximum x and y coordinates to get the bounding box
        min_x, min_y = np.min(non_zero_coords, axis=0)
        max_x, max_y = np.max(non_zero_coords, axis=0)

        # Add padding if needed
        padding = 0
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img_p.size[0], max_x + padding)
        max_y = min(img_p.size[1], max_y + padding)

        # The bounding box coordinates are now (min_x, min_y, max_x, max_y)
        # Scale the bounding box coordinates to match the image size (1024x1024)
        min_x *= 2
        min_y *= 2
        max_x *= 2
        max_y *= 2
        bbox = [min_x, min_y, max_x, max_y]
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
        # img_p, label, label_vis = self.load_single_image(index)
        # bbox=[30,50,60,100]
   
        ### Get reference image
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        # save_image(image_tensor_resize, "Train_data_images/"+str(index)+'_image_tensor_resize.png')
        # save_image(inpaint_tensor_resize, "Train_data_images/"+ str(index)+'_inpaint_tensor_resize.png')
        # save_image(mask_tensor_resize, "Train_data_images/"+ str(index)+'_mask_tensor_resize.png')
        # save_image(ref_image_tensor,  "Train_data_images/"+str(index)+'_ref_image_tensor.png')
        
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}


    def __len__(self):
        return self.length
    
    



        





    
    



        






