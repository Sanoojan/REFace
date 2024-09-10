import argparse
import gc
import random
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from dift.src.models.dift_sd import SDFeaturizer
# from dift.src.utils.visualization import Demo

from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn as nn

cos = nn.CosineSimilarity(dim=0)
import numpy as np  

dift = SDFeaturizer()
prompt = f'a photo of a human face'
# decrease these two if you don't have enough RAM or GPU memory
img_size = 512
ensemble_size = 8
t=261
save_path='intermediate_results_FFHQ_261'

#make a folder to save results
import os
if not os.path.exists(save_path+"/results"):
    os.makedirs(save_path+"/results")

Dataset='FFHQ'

if Dataset=='CelebA-HQ':
    # This is for CelebA-HQ
    src_start=29000
    tar_start=28000

elif Dataset=='FFHQ':
    # This is for FFHQ
    src_start=69000
    tar_start=68000

for im in tqdm(range(1000)):
    src_id=im+src_start
    tar_id=im+tar_start
    
    if Dataset=='CelebA-HQ':
        # This is for CelebA-HQ
        filelist = ['dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img/'+str(tar_id) +'.jpg', 'dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img/'+str(src_id) +'.jpg']
        source_mask_path='dataset/FaceData/CelebAMask-HQ/CelebA-HQ-mask/14/'+str(tar_id) +'_skin.png'
    elif Dataset=='FFHQ':
        # This is for FFHQ
        filelist = ['dataset/FaceData/FFHQ/Val_target/'+str(tar_id) +'.png', 'dataset/FaceData/FFHQ/Val/'+str(src_id) +'.png']
        source_mask_path='dataset/FaceData/FFHQ/target_mask/'+str(tar_id) +'.png'
    
    
    
    
    ft = []
    imglist = []


    for filename in filelist:
        img = Image.open(filename).convert('RGB')
        img = img.resize((img_size, img_size))
        imglist.append(img)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        ft.append(dift.forward(img_tensor,
                            prompt=prompt,t=t,
                            ensemble_size=ensemble_size))
    ft = torch.cat(ft, dim=0)
    num_channel = ft.size(1)
    # gc.collect()
    # torch.cuda.empty_cache()

    
    mask=Image.open(source_mask_path).convert('L')
    mask=mask.resize((img_size,img_size))
    if Dataset=='FFHQ':
        preserve = [1,2,3,5,6,7,8,9]
        # if the mask is not in the preserve list, then make it 0
        mask=np.array(mask)
        mask_im = np.isin(mask, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask)
        converted_mask[mask_im] = 255
        # convert to PIL image
        mask=Image.fromarray(converted_mask).convert('L')
        
    # convert mask to torch tensor
    mask_tensor = (PILToTensor()(mask) / 255.0) 
    
    
        
    
    # find the indices where the mask is 1
    mask_indices_or = torch.nonzero(mask_tensor[0])
    # breakpoint()
    img_tensors=[]
    for im in imglist:
        img_tensors.append((PILToTensor()(im) / 255.0 - 0.5) * 2)
        


    src_ft = ft[0].unsqueeze(0)
    src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)          
    trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft[1:])

    # target_mask_ind=mask_indices   # like mask_indices


    # src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
    # cos_map = cos(src_ft, trg_ft).cpu().numpy()  # N, H, W

    src_ft2=src_ft.squeeze(0).reshape(num_channel,-1)
    tar_ft2=trg_ft.squeeze(0).reshape(num_channel,-1)
    # cos_map2=cos(src_ft2,tar_ft2).cpu().numpy()
    src_ft2=torch.nn.functional.normalize(src_ft2, p=2, dim=0)
    tar_ft2=torch.nn.functional.normalize(tar_ft2, p=2, dim=0)
    # chunk the tar_ft2 into 8 chunks
    src_ft2=torch.chunk(src_ft2,32,dim=1)
    # breakpoint()
    cos_matr=[]
    for i in range(32):
        cos_matrix= torch.mm(src_ft2[i].T,tar_ft2)
        cos_matrix=torch.argmax(cos_matrix,dim=1).cpu().numpy()
        cos_matr.append(cos_matrix)
    # cos_matrix=torch.mm(src_ft2.T,tar_ft2)
    # find argmax
    # cos_matrix=torch.argmax(cos_matrix,dim=1).cpu().numpy()
    
    cos_matrix=np.concatenate(cos_matr,axis=0)
    # breakpoint()



    src_tensor=img_tensors[0]
    tar_tensor=img_tensors[1]
    src_tensor=torch.reshape(src_tensor,(3,-1))
    tar_tensor=torch.reshape(tar_tensor,(3,-1))
    # change mask_indices_or to flatten indices
    mask_indices_or=mask_indices_or[:,0]*img_size+mask_indices_or[:,1]

    # change the tar_tensor at the mask_indices_or from src_tensor for corresponding indices from target_mask_ind

    # 
    for i in range(len(mask_indices_or)):
        ind=mask_indices_or[i]

        src_tensor[:,ind]=tar_tensor[:,cos_matrix[ind]]   
        
    # tar_tensor=torch.reshape(tar_tensor,(3,img_size,img_size))
    # visualize tar_tensor
    src_tensor_show=torch.reshape(src_tensor,(3,img_size,img_size))
    # tar_tensor=tar_tensor.squeeze(0)
    src_tensor_show=src_tensor_show.permute(1,2,0)
    src_tensor_show=src_tensor_show.numpy()
    src_tensor_show=src_tensor_show/2+0.5
    # plt.imshow(tar_tensor_show)
    # plt.show()
    #save the tar_tensor as image
    
    plt.imsave(save_path+"/results/"+str(tar_id).zfill(12)+".jpg",src_tensor_show) 