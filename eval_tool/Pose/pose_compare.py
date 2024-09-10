"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
import re
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F
# from src.Face_models.encoders.model_irse import Backbone
import eval_tool.face_vid2vid.modules.hopenet as hopenet1
from torchvision import models
# import clip
import torchvision
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
from natsort import natsorted
# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    default=['/share/data/drive_3/Sanoojan/needed/Paint_for_swap/dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # _, self.preprocess = clip.load("ViT-B/32", device=device)
        # self.preprocess
        # eval_transform = transforms.Compose([transforms.ToTensor(),
        #                                  transforms.Resize(112),
        #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_hopenet =  torchvision.transforms.Compose([TF.ToTensor(),TF.Resize(size=(224, 224)),
                                                     TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        image = self.transform_hopenet(Image.open(path).convert('RGB'))
        return image

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def compute_features(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files),3))

    start_idx = 0
    
   
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        # breakpoint()
        with torch.no_grad():
            # x = face_pool_1(batch)  if batch.shape[2]!=256 else  batch # (1) resize to 256 if needed
            # x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
            # x = face_pool_2(x) # (3) resize to 112 to fit pre-trained model
            # breakpoint()
            yaw_gt, pitch_gt, roll_gt = model(batch )
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)
            # breakpoint()
        # breakpoint()
        # # If model output is not scalar, apply global spatial average pooling.
        # # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        # pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        yaw_gt = yaw_gt.cpu().numpy().reshape(-1,1)
        pitch_gt = pitch_gt.cpu().numpy().reshape(-1,1)
        roll_gt = roll_gt.cpu().numpy().reshape(-1,1)
        # breakpoint()
        pred_arr[start_idx:start_idx + yaw_gt.shape[0]] = np.concatenate((yaw_gt,pitch_gt,roll_gt),axis=1)

        start_idx = start_idx + yaw_gt.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_features_wrapp(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = natsorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        
        # Extract all numbers before the dot using regular expression
        # breakpoint()
        pattern = r'[_\/.-]'

        # Split the file path using the pattern
        parts = [re.split(pattern, str(file.name)) for file in files]
        # breakpoint()
        # Filter out non-numeric parts and convert to integers
        numbers =[[int(par) for par in part if par.isdigit()] for part in parts]
        
        numbers= [ num[-1] for num in numbers if len(num)>0]
        mi_num= min(numbers)
        # breakpoint()
        # if numbers[0]>28000: # CelebA-HQ Test my split #check 28000-29000: target 29000-30000: source
        numbers = [(num - mi_num) for num in numbers] # celeb
        # breakpoint()
        # if numbers[0]>27500 and numbers[0]<30000: # CelebA-HQ Test my split #check 28000-29000: target 29000-30000: source
        #     numbers = [num-28000 for num in numbers] # celeb
        # else:
        #     numbers = [num-68000 for num in numbers]
        
        pred_arr = compute_features(files, model, batch_size,
                                               dims, device, num_workers)

    return pred_arr,numbers


def calculate_id_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx]).to(device)
    
    hopenet = hopenet1.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    print('Loading hopenet')
    hopenet_state_dict = torch.load('eval_tool/Face_rec_models/hopenet_robust_alpha1.pkl')
    hopenet.load_state_dict(hopenet_state_dict)
    if torch.cuda.is_available():
        hopenet = hopenet.cuda()
        hopenet.eval()

    feat1,ori_lab = compute_features_wrapp(paths[0], hopenet, batch_size,
                                        dims, device, num_workers)
    feat2,swap_lab = compute_features_wrapp(paths[1], hopenet, batch_size,
                                        dims, device, num_workers)
    
    # breakpoint()
    # top5 = np.sum(np.isin(np.argsort(dot_prod,axis=1)[:,-5:],swap_lab))/len(swap_lab)
    # breakpoint()
    feat1=feat1[swap_lab]
    # find l2 distance
    dist = np.linalg.norm(feat1-feat2,axis=1)
    Value= np.mean(dist)
    
    return Value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    Pose_value= calculate_id_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          2048,
                                          num_workers)
    print('Pose_value: ',Pose_value)


if __name__ == '__main__':
    main()