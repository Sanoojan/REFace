from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os 
from os import path as osp
from io import BytesIO
import json
import logging
import base64
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
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
import clip
import bezier


celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']
# 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow, 8:l_ear, 9:r_ear, 
# 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat, 15:ear_r, 16:neck_l, 17:neck, 18:cloth
# preserve=[1,2,4,5,8,9,17 ] #comes from source
preserve = [1,2,4,5,8,9, 6,7,10,11,12]

faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                    'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']
#FFHQ/ faceparcing network
# 0:background, 1:lip, 2:eyebrows, 3:eyes, 4:hair, 5:nose, 6:skin, 7:ears, 
# 8:belowface, 9:mouth, 10:eye_glass, 11:ear_rings
preserve = [1,2,5,6,7,9]
# 2,5,6,8,
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








class VideoDataset(data.Dataset):
    def __init__(self,label_transform=None,data_path='Video_processing/target_frames',mask_path='Video_processing/target_masks' ,**args
        ):
        self.label_transform=label_transform
        self.args=args
        self.load_prior=False
        self.kernel = np.ones((1, 1), np.uint8)
        self.gray_outer_mask=True
        
        self.Fullmask=False
        self.bbox_path_list=[]
        
        self.data_path=data_path
        self.mask_path=mask_path
        self.gray_outer_mask=args['gray_outer_mask']
        
        # print(args)
        # breakpoint()
        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['remove_mask_tar_FFHQ']
            self.remove_tar=args['remove_mask_tar_FFHQ']
            self.preserve_src=args['preserve_mask_src_FFHQ']
    
        self.Fullmask=False
        # get all imgs in data_path
        self.imgs = [osp.join(data_path, str(img)+".png") for img in range(len(os.listdir(data_path)))]
        # print(self.imgs)
        self.labels = [osp.join(mask_path, str(img)+".png") for img in range(len(os.listdir(mask_path)))]
        
        assert len(self.imgs) == len(self.labels), "The number of images must be equal to the number of labels"
        
        # self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 29000)])
        # self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 29000)]) 
        
        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)
        # self.preserve=preserve
    
    def __getitem__(self, index):
        if self.gray_outer_mask:
            return self.__getitem_gray__(index)
        else:
            return self.__getitem_black__(index)

    def __getitem_gray__(self, index):
        # uses the black mask in reference
        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB').resize((512,512))
        # if self.img_transform is not None:
        #     img = self.img_transform(img)

        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        
        if self.Fullmask:
            mask_img_full=mask_img
            mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9, 6,7,10,11,12]
        preserve=self.remove_tar
        # preserve=[2,3,5,6,7] 
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')

   

        ### Get reference
        # ref_img_path = self.ref_imgs[index]
        # img_p_np=cv2.imread(ref_img_path)
        # # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        # ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # # ref_img= cv2.resize(ref_img, (224, 224))
        
        # ref_mask_path = self.ref_labels[index]
        # ref_mask_img = Image.open(ref_mask_path).convert('L')
        # ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        # preserve=[1,2,4,5,8,9 ,6,7,10,11,12,17 ]
        # preserve=self.preserve
        # preserve = [1,2,3,5,6,7,9]
        # ref_mask= np.isin(ref_mask_img, preserve)

        # # Create a converted_mask where preserved values are set to 255
        # ref_converted_mask = np.zeros_like(ref_mask_img)
        # ref_converted_mask[ref_mask] = 255
        # ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        # ref_mask_img=Image.fromarray(ref_img).convert('L')
        
        
        # ref_img=self.trans(image=ref_img)
        # ref_img=Image.fromarray(ref_img["image"])
        # ref_img=get_tensor_clip()(ref_img)
        
        # ref_mask_img_r = ref_converted_mask.resize(ref_img.shape[1::], Image.NEAREST)
        # ref_mask_img_r = np.array(ref_mask_img_r)
        # ref_img=ref_img*ref_mask_img_r
        # ref_img[ref_mask_img_r==0]=0
        
        # ref_img=Image.fromarray(ref_img)
        
        # ref_img=get_tensor_clip()(ref_img)
        
        
        


        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

   

        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
        # reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        inpaint_tensor=image_tensor*mask_tensor
        
        # mask_ref=T.Resize((224,224))(reference_mask_tensor)
   
        # breakpoint()
        # ref_img=ref_img*mask_ref   # comment here if you want the full ref img
        # ref_image_tensor = ref_img.unsqueeze(0)
        
        if self.load_prior:
            prior_img_path = self.prior_images[index]
            prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
            prior_image_tensor=get_tensor()(prior_img)
            # prior_image_tensor = prior_img
        else:
            prior_image_tensor = image_tensor
        
        if self.Fullmask:
            return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor},str(index).zfill(12)
    
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor},str(index).zfill(12)
        
        
    
    def __getitem_black__(self, index):
        # uses the black mask in reference
        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB').resize((512,512))
        
        # # create a img with full white 
        # img_p_np=np.array(img_p)
        # img_p_np[:,:,:]=255
        # img_p_white=Image.fromarray(img_p_np).convert('RGB')
        # img_p_white=get_tensor()(img_p_white)

        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        preserve = [2,3,5,6,7]  # face parser mapping
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')

   
        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

   

        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        

        inpaint_tensor=image_tensor*mask_tensor
        
        if self.load_prior:
            prior_img_path = self.prior_images[index]
            prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
            prior_image_tensor=get_tensor()(prior_img)
            # prior_image_tensor = prior_img
        else:
            prior_image_tensor = image_tensor
    
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor},str(index).zfill(12)
  

    def __len__(self):
        return self.length
    
    
    


    
    # def __getitem__(self, index):
    #     # uses the gray mask in reference
        
    #     img_path = self.imgs[index]
    #     img_p = Image.open(img_path).convert('RGB').resize((512,512))
    #     # if self.img_transform is not None:
    #     #     img = self.img_transform(img)

    #     mask_path = self.labels[index]
    #     mask_img = Image.open(mask_path).convert('L')
        
    #     if self.Fullmask:
    #         mask_img_full=mask_img
    #         mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
    #     mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

    #     # Create a mask to preserve values in the 'preserve' list
    #     # preserve = [1,2,4,5,8,9,17 ]
    #     # preserve = [1,2,4,5,8,9, 6,7,10,11,12]
    #     preserve=[1,2,4,5,8,9, 6,7,10,11,12,13,17]
    #     mask = np.isin(mask_img, preserve)

    #     # Create a converted_mask where preserved values are set to 255
    #     converted_mask = np.zeros_like(mask_img)
    #     converted_mask[mask] = 255
    #     # convert to PIL image
    #     mask_img=Image.fromarray(converted_mask).convert('L')

   

    #     ### Get reference
    #     ref_img_path = self.ref_imgs[index]
    #     img_p_np=cv2.imread(ref_img_path)
    #     # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
    #     ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
    #     # ref_img= cv2.resize(ref_img, (224, 224))
        
    #     ref_mask_path = self.ref_labels[index]
    #     ref_mask_img = Image.open(ref_mask_path).convert('L')
    #     ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

    #     # Create a mask to preserve values in the 'preserve' list
    #     # preserve = [1,2,4,5,8,9,17 ]
    #     # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
    #     preserve=[1,2,4,5,8,9, 6,7,10,11,12,13,17]
    #     # preserve = [1,2,4,5,8,9 ]
    #     ref_mask= np.isin(ref_mask_img, preserve)

    #     # Create a converted_mask where preserved values are set to 255
    #     ref_converted_mask = np.zeros_like(ref_mask_img)
    #     ref_converted_mask[ref_mask] = 255
    #     ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
    #     # convert to PIL image
        
        
    #     # ref_mask_img=Image.fromarray(ref_img).convert('L')
        
        
    #     ref_img=self.trans(image=ref_img)
    #     ref_img=Image.fromarray(ref_img["image"])
    #     ref_img=get_tensor_clip()(ref_img)
        
    #     # ref_mask_img_r = ref_converted_mask.resize(ref_img.shape[1::], Image.NEAREST)
    #     # ref_mask_img_r = np.array(ref_mask_img_r)
    #     # ref_img=ref_img*ref_mask_img_r
    #     # ref_img[ref_mask_img_r==0]=0
        
    #     # ref_img=Image.fromarray(ref_img)
        
    #     # ref_img=get_tensor_clip()(ref_img)
        
        
        


    #     ### Crop input image
    #     image_tensor = get_tensor()(img_p)
    #     W,H = img_p.size

   

    #     mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
    #     reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
    #     inpaint_tensor=image_tensor*mask_tensor
        
    #     mask_ref=T.Resize((224,224))(reference_mask_tensor)
   
    #     # breakpoint()
    #     # ref_img=ref_img*mask_ref   # comment here if you want the full ref img
    #     ref_image_tensor = ref_img.unsqueeze(0)
        
    #     if self.load_prior:
    #         prior_img_path = self.prior_images[index]
    #         prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
    #         prior_image_tensor=get_tensor()(prior_img)
    #         # prior_image_tensor = prior_img
    #     else:
    #         prior_image_tensor = None
        
    #     if self.Fullmask:
    #         return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor},str(index).zfill(12)
    
    #     return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)
        
        
    
    