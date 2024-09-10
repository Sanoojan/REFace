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
preserve = [1,2,3,5,6,7,9]
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


class COCOImageDataset(data.Dataset):
    def __init__(self,test_bench_dir):

        self.test_bench_dir=test_bench_dir
        self.id_list=np.load('test_bench/id_list.npy')
        self.id_list=self.id_list.tolist()
        print("length of test bench",len(self.id_list))
        self.length=len(self.id_list)

       

    
    def __getitem__(self, index):
        img_path=os.path.join(os.path.join(self.test_bench_dir,'GT_3500',str(self.id_list[index]).zfill(12)+'_GT.png'))
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'Ref_3500',str(self.id_list[index]).zfill(12)+'_ref.png'))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_img=get_tensor_clip()(ref_img)
        ref_image_tensor = ref_img.unsqueeze(0)


        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

   
        ### bbox mask
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'Mask_bbox_3500',str(self.id_list[index]).zfill(12)+'_mask.png'))
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)



      

        inpaint_tensor=image_tensor*mask_tensor
    
        return image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(self.id_list[index]).zfill(12)



    def __len__(self):
        return self.length





class CelebAdataset(data.Dataset):
    def __init__(self,state,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.load_prior=True
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
            ])
        self.trans=A.Compose([
            A.Resize(height=224,width=224)])
        
        self.gray_outer_mask=args['gray_outer_mask']
        
        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['preserve_mask_src']
            self.remove_tar=args['remove_mask_tar']
            self.preserve_src=args['preserve_mask_src']
        
        self.Fullmask=False
        self.bbox_path_list=[]
        if state == "train":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000)])
            # self.labels = ([os.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000)]) if self.load_vis_img else None
        elif state == "validation":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None
        else:
            data_path="intermediate_renact/results_2"
            ref_img_path="dataset/FaceData/CelebAMask-HQ/Val_cropped"
            ref_img_mask_path="dataset/FaceData/CelebAMask-HQ/Val_cropped_mask"
            # self.imgs=sorted([osp.join(data_path, "%d.jpg"%idx) for idx in range(28000, 29000)])
            
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 29000)])
            

            
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 29000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 29000)]) if self.load_vis_img else None

            self.ref_imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(29000, 30000)])
            # self.ref_imgs = sorted([osp.join(ref_img_path, "%d.jpg"%idx) for idx in range(29000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.ref_labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(29000, 30000)]) 
            # self.ref_labels =  sorted([osp.join(ref_img_mask_path, "%d.png"%idx) for idx in range(29000, 30000)]) 
            self.ref_abels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(29000, 30000)]) if self.load_vis_img else None


            # self.ref_imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 29000)])
            # self.ref_labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 29000)]) 
            # self.ref_abels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 29000)]) if self.load_vis_img else None

            self.ref_imgs= self.ref_imgs[:int(len(self.imgs)*self.fraction)]
            self.ref_labels= self.ref_labels[:int(len(self.labels)*self.fraction)]
            self.ref_labels_vis= self.ref_labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None
            # intermediate_results_261/results/000000028000.jpg
            if self.load_prior:
                # self.prior_images=sorted([osp.join("intermediate_results_261/results", "0000000%d.jpg"%idx) for idx in range(28000, 29000)])
                self.prior_images=sorted([osp.join("intermediate_renact/results_2", "%d.jpg"%idx) for idx in range(28000, 29000)])
        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]
        self.labels_vis= self.labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding 
            - the image, 
            - the semantic image, 
            - the corresponding visualization image

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
            label: seg mask
            label_vis: visualization of the seg mask
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        # print(label)
        label = Image.open(label).convert('L')
        # breakpoint()
        # label2=TO_TENSOR(label)
        # save_image(label2, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        # save_image(label, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  

        return img, label, label_vis
  

    
    
    def __getitem__(self, index):
        if self.gray_outer_mask:
            return self.__getitem_gray__(index)
        else:
            return self.__getitem_black__(index)
    
    def __getitem_gray__(self, index):
        # uses the gray mask in reference
        
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
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')

   

        ### Get reference
        ref_img_path = self.ref_imgs[index]
        img_p_np=cv2.imread(ref_img_path)
        # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # ref_img= cv2.resize(ref_img, (224, 224))
        
        ref_mask_path = self.ref_labels[index]
        ref_mask_img = Image.open(ref_mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        # preserve=[1,2,4,5,8,9 ,6,7,10,11,12,13,17 ]
        preserve=self.preserve_src
        # preserve=self.preserve
        # preserve = [1,2,4,5,8,9 ]
        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        # ref_mask_img=Image.fromarray(ref_img).convert('L')
        
        
        ref_img=self.trans(image=ref_img)
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor_clip()(ref_img)
        
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
        reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        inpaint_tensor=image_tensor*mask_tensor
        
        mask_ref=T.Resize((224,224))(reference_mask_tensor)
   
        # breakpoint()
        ref_img=ref_img*mask_ref   # comment here if you want the full ref img
        ref_image_tensor = ref_img.unsqueeze(0)
        
        if self.load_prior:
            prior_img_path = self.prior_images[index]
            prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
            prior_image_tensor=get_tensor()(prior_img)
            # prior_image_tensor = prior_img
        else:
            prior_image_tensor = None
        
        if self.Fullmask:
            return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor},str(index).zfill(12)
    
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)
        
        
    
    def __getitem_black__(self, index):
        # uses the black mask in reference
        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB').resize((512,512))
 

        mask_path = self.labels[index]
        mask_img = Image.open(mask_path).convert('L')
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        preserve = self.preserve
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')

   

        ### Get reference
        ref_img_path = self.ref_imgs[index]
        img_p_np=cv2.imread(ref_img_path)
        # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # ref_img= cv2.resize(ref_img, (224, 224))
        
        ref_mask_path = self.ref_labels[index]
        ref_mask_img = Image.open(ref_mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        preserve = self.preserve
        # preserve = [1,2,4,5,8,9 ]
        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        ref_mask_img=Image.fromarray(ref_img).convert('L')
        ref_mask_img_r = ref_converted_mask.resize(img_p_np.shape[1::-1], Image.NEAREST)
        ref_mask_img_r = np.array(ref_mask_img_r)
        # ref_img[ref_mask_img_r==0]=0
        
        ref_img=self.trans(image=ref_img)
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor_clip()(ref_img)
        
        ref_image_tensor = ref_img.unsqueeze(0)
        
        


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
            prior_image_tensor = None
    
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)
  

    def __len__(self):
        return self.length
    
    
    
class FFHQdataset(data.Dataset):
    def __init__(self,state,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.load_prior=False
        self.kernel = np.ones((1, 1), np.uint8)
        self.Fullmask=False
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
            ])
        self.trans=A.Compose([
            A.Resize(height=224,width=224)])
        
        self.bbox_path_list=[]
        
        self.gray_outer_mask=args['gray_outer_mask']
        
        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['preserve_mask_src_FFHQ']
            self.remove_tar=args['remove_mask_tar_FFHQ']
            self.preserve_src=args['preserve_mask_src_FFHQ']
        
        
        
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

            if self.load_prior:
                self.prior_images=sorted([osp.join("intermediate_results_FFHQ_261/results", "0000000%d.jpg"%idx) for idx in range(68000, 69000)])
            
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
        # uses the gray mask in reference
        
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
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        

        ### Get reference
        ref_img_path = self.ref_imgs[index]
        img_p_np=cv2.imread(ref_img_path)
        # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # ref_img= cv2.resize(ref_img, (224, 224))
        
        ref_mask_path = self.ref_labels[index]
        ref_mask_img = Image.open(ref_mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        # preserve=[1,2,4,5,8,9 ,6,7,10,11,12,13,17 ]
        preserve=self.preserve_src
        # preserve=self.preserve
        # preserve = [1,2,4,5,8,9 ]
        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        # ref_mask_img=Image.fromarray(ref_img).convert('L')
        
        
        ref_img=self.trans(image=ref_img)
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor_clip()(ref_img)
        
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
        reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        inpaint_tensor=image_tensor*mask_tensor
        
        mask_ref=T.Resize((224,224))(reference_mask_tensor)
   
        
        ref_img=ref_img*mask_ref   # comment here if you want the full ref img
        ref_image_tensor = ref_img.unsqueeze(0)
        
        if self.load_prior:
            prior_img_path = self.prior_images[index]
            prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
            prior_image_tensor=get_tensor()(prior_img)
            # prior_image_tensor = prior_img
        else:
            prior_image_tensor = image_tensor
        
        if self.Fullmask:
            return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor},str(index).zfill(12)

        
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)
        

    def __len__(self):
        return self.length
    
    
    
class FFdataset(data.Dataset):
    def __init__(self,state,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.load_prior=False
        self.kernel = np.ones((1, 1), np.uint8)
        self.Fullmask=False
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
            ])
        self.trans=A.Compose([
            A.Resize(height=224,width=224)])
        
        self.bbox_path_list=[]
        
        self.gray_outer_mask=args['gray_outer_mask']
        
        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['preserve_mask_src_FFHQ']
            self.remove_tar=args['remove_mask_tar_FFHQ']
            self.preserve_src=args['preserve_mask_src_FFHQ']
        
        if state == "train":
            self.imgs = sorted([osp.join(args['dataset_dir'], "images512", '{0:0=5d}.png'.format(idx)) for idx in range(68000)])
            self.labels = sorted([osp.join(args['dataset_dir'], "BiSeNet_mask",'{0:0=5d}.png'.format(idx))  for idx in range(68000)])

        elif state == "validation":
            self.imgs = sorted([osp.join(args['dataset_dir'], "images512", '{0:0=5d}.png'.format(idx)) for idx in range(68000,70000)])
            self.labels = sorted([osp.join(args['dataset_dir'], "BiSeNet_mask",'{0:0=5d}.png'.format(idx))  for idx in range(68000,70000)])
            
        else:
            self.imgs = sorted([osp.join(args['dataset_dir'], "Val_target",'{0:0=4d}.png'.format(idx))  for idx in range(0,500)])
            self.labels = sorted([osp.join(args['dataset_dir'], "target_mask", '{0:0=4d}.png'.format(idx))  for idx in range(0,500)]) 
            

            self.ref_imgs = sorted([osp.join(args['dataset_dir'], "Val", '{0:0=4d}.png'.format(idx))  for idx in range(500,1000)])
            self.ref_labels =  sorted([osp.join(args['dataset_dir'], "src_mask", '{0:0=4d}.png'.format(idx))  for idx in range(500,1000)]) 
            

            self.ref_imgs= self.ref_imgs[:int(len(self.imgs)*self.fraction)]
            self.ref_labels= self.ref_labels[:int(len(self.labels)*self.fraction)]

            if self.load_prior:
                self.prior_images=sorted([osp.join("intermediate_results_FFHQ_261/results", "0000000%d.jpg"%idx) for idx in range(68000, 69000)])
            
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
        # uses the gray mask in reference
        
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
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L')
        

        ### Get reference
        ref_img_path = self.ref_imgs[index]
        img_p_np=cv2.imread(ref_img_path)
        # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # ref_img= cv2.resize(ref_img, (224, 224))
        
        ref_mask_path = self.ref_labels[index]
        ref_mask_img = Image.open(ref_mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        # preserve=[1,2,4,5,8,9 ,6,7,10,11,12,13,17 ]
        preserve=self.preserve_src
        # preserve=self.preserve
        # preserve = [1,2,4,5,8,9 ]
        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        # ref_mask_img=Image.fromarray(ref_img).convert('L')
        
        
        ref_img=self.trans(image=ref_img)
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor_clip()(ref_img)
        
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
        reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        inpaint_tensor=image_tensor*mask_tensor
        
        mask_ref=T.Resize((224,224))(reference_mask_tensor)
   
        
        ref_img=ref_img*mask_ref   # comment here if you want the full ref img
        ref_image_tensor = ref_img.unsqueeze(0)
        
        if self.load_prior:
            prior_img_path = self.prior_images[index]
            prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
            prior_image_tensor=get_tensor()(prior_img)
            # prior_image_tensor = prior_img
        else:
            prior_image_tensor = image_tensor
        
        if self.Fullmask:
            return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_img_full,"ref_imgs":ref_image_tensor},str(index).zfill(12)

        
        return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)
     
    #  def __getitem__(self, index):
    #     # uses the black mask in reference
        
    #     img_path = self.imgs[index]
    #     img_p = Image.open(img_path).convert('RGB').resize((512,512))


    #     mask_path = self.labels[index]
    #     mask_img = Image.open(mask_path).convert('L').resize((512,512))
    #     mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

    #     # Create a mask to preserve values in the 'preserve' list
        
    #     preserve = [1,2,3,5,6,7,9] #FFHQ
        
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
    #     preserve = [1,2,3,5,6,7,9] #FFHQ
    #     ref_mask= np.isin(ref_mask_img, preserve)

    #     # Create a converted_mask where preserved values are set to 255
    #     ref_converted_mask = np.zeros_like(ref_mask_img)
    #     ref_converted_mask[ref_mask] = 255
    #     ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
    #     # convert to PIL image
        
        
    #     ref_mask_img=Image.fromarray(ref_img).convert('L')
    #     ref_mask_img_r = ref_converted_mask.resize(img_p_np.shape[1::-1], Image.NEAREST)
    #     ref_mask_img_r = np.array(ref_mask_img_r)
    #     ref_img[ref_mask_img_r==0]=0
        
    #     ref_img=self.trans(image=ref_img)
    #     ref_img=Image.fromarray(ref_img["image"])
    #     ref_img=get_tensor_clip()(ref_img)
        
    #     ref_image_tensor = ref_img.unsqueeze(0)
        
        


    #     ### Crop input image
    #     image_tensor = get_tensor()(img_p)
    #     W,H = img_p.size



    #     mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

    #     inpaint_tensor=image_tensor*mask_tensor
        
    #     if self.load_prior:
    #         prior_img_path = self.prior_images[index]
    #         prior_img = Image.open(prior_img_path).convert('RGB').resize((512,512))
    #         prior_image_tensor=get_tensor()(prior_img)
    #         # prior_image_tensor = prior_img
    #     else:
    #         prior_image_tensor = image_tensor
    
    #     return image_tensor,prior_image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(index).zfill(12)


    def __len__(self):
        return self.length