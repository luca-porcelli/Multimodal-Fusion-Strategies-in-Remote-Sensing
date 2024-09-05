import os
import torch
import random
from glob import glob
import numpy as np
from tqdm import tqdm
from osgeo import gdal

import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset

import albumentations as A

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Pixel-Level class distribution (total sum equals 1.0)
#class_distr = torch.Tensor([0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963])
class_distr = torch.Tensor([0.000947, 0.044522, 0.000590, 0.000035, 0.000208, 0.005150, 
 0.792127, 0.008515, 0.001689, 0.103776, 0.029676, 0.000183, 0.000222, 0.000091, 0.012270])

bands_mean = np.array([0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902, 0.03680401, 0.03999107, 0.03566642, 0.03965081, 0.0267993,  0.01978944]).astype('float32')
bands_std = np.array([0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412, 0.04992323, 0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115]).astype('float32')

bands_mean_10 = np.array([0.0582676,  0.05223386, 0.04381474, 0.0357083]).astype('float32')
bands_std_10 = np.array([0.03240627, 0.03432253, 0.0354812,  0.03757695]).astype('float32')

bands_mean_20 = np.array([0.03412902, 0.03680401, 0.03999107, 0.03566642, 0.03965081, 0.0267993]).astype('float32')
bands_std_20 = np.array([0.03785412, 0.04992323, 0.05884482, 0.05545856, 0.06423746, 0.04211187]).astype('float32')

bands_mean_60 = np.array([0.01978944]).astype('float32')
bands_std_60 = np.array([0.03019115]).astype('float32')

# MADOS DATASET
def get_band(path):
    return int(path.split('_')[-2])

class MADOS(Dataset): # Extend PyTorch's Dataset class
    def __init__(self, path, splits, mode = 'train'):
        
        if mode=='train':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'train_X.txt'),dtype='str')
                
        elif mode=='test':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'test_X.txt'),dtype='str')
                
        elif mode=='val':
            self.ROIs_split = np.genfromtxt(os.path.join(splits, 'val_X.txt'),dtype='str')
            
        else:
            raise
        self.X = []           # Loaded Images
        self.y = []           # Loaded Output masks
            
        self.tiles = glob(os.path.join(path,'*'))

        for tile in tqdm(self.tiles, desc = 'Load '+mode+' set to memory'):

                # Get the number of different crops for the specific tile
                splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]
                
                for crop in splits:
                    crop_name = os.path.basename(tile)+'_'+crop.split('.tif')[0]
                    
                    if crop_name in self.ROIs_split:
    
                        # Load Input Images
                        # Get the bands for the specific crop 
                        all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_'+crop))
                        all_bands = sorted(all_bands, key=get_band)

            
                        ################################
                        # Upsample the bands #
                        ################################
                        current_image = []
                        for c, band in enumerate(all_bands, 1):
                            upscale_factor = int(os.path.basename(os.path.dirname(band)))//10
            
                            with rasterio.open(band, mode ='r') as src:
                                current_image.append(src.read(1,
                                                                out_shape=(
                                                                    int(src.height * upscale_factor),
                                                                    int(src.width * upscale_factor)
                                                                ),
                                                                resampling=Resampling.nearest
                                                              ).copy()
                                                  )
                        
                        stacked_image = np.stack(current_image)
                        self.X.append(stacked_image)
                        

                        # Load Classsification Mask
                        cl_path = os.path.join(tile,'10',os.path.basename(tile)+'_L2R_cl_'+crop)
                        ds = gdal.Open(cl_path)
                        temp = np.copy(ds.ReadAsArray().astype(np.int64))
                        
                        ds=None # Close file
                        self.y.append(temp)
            
            
        self.X = np.stack(self.X)
        self.y = np.stack(self.y)
        
        # Categories from 1 to 0
        self.y = self.y - 1

        self.impute_nan = np.tile(bands_mean, (self.X.shape[-1],self.X.shape[-2],1))
        self.mode = mode
        self.length = len(self.y)
        self.path = path
        self.input_size = 240
        
    def __len__(self):
        return self.length
    
    def getnames(self):
        return self.ROIs_split
    
    def __getitem__(self, index):
        image = self.X[index]
        target = self.y[index]

        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')       # CxWxH to WxHxC
        
        nan_mask = np.isnan(image)
        image[nan_mask] = self.impute_nan[nan_mask]
        
        target = target[:,:,np.newaxis]
        
        image, target = self.transform(image, target)
        
        if self.mode=='train':
            image, target = self.join_transform_old(image, target)

        #image = (image.astype(np.float32).transpose(2, 0, 1).copy() - bands_mean.reshape(-1,1,1))/ bands_std.reshape(-1,1,1)
        image10 = (image[:, :, :4].astype(np.float32).transpose(2, 0, 1).copy() - bands_mean_10.reshape(-1,1,1))/ bands_std_10.reshape(-1,1,1)
        image20 = (image[:, :, 4:10].astype(np.float32).transpose(2, 0, 1).copy() - bands_mean_20.reshape(-1,1,1))/ bands_std_20.reshape(-1,1,1)
        image60 = (image[:, :, 10:].astype(np.float32).transpose(2, 0, 1).copy() - bands_mean_60.reshape(-1,1,1))/ bands_std_60.reshape(-1,1,1)
        target = target.squeeze()
        
        return image10.copy(), image20.copy(), image60.copy(), target.copy()
    
    def transform(self, image, target):
        aug = A.RandomCrop(width=224, height=224)
        random.seed(7) 
        augmented = aug(image=image, mask=target) 
        image_scaled = augmented['image'] 
        mask_scaled = augmented['mask'] 
        return image_scaled, mask_scaled
    '''
    def join_transform(self, image, target):
        aug = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5)
            ], p=1),
            A.Rotate(limit=90, p=0.8)
        ])
        augmented = aug(image=image, mask=target)
        image_augmented = augmented['image']
        target_augmented = augmented['mask']
        return image_augmented, target_augmented
    '''
    def join_transform_old(self, image, target):
        # Random Flip image
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
        if f != 2:
            image = self.filp_array(image, f)
            target = self.filp_array(target,f)
             
        # Random Rotate (Only 0, 90, 180, 270)
        if np.random.random() < 0.8:
            k = np.random.randint(0, 4)  # [0, 1, 2, 3]
            image = np.rot90(image, k, (1, 0))  # clockwise
            target = np.rot90(target, k, (1, 0))
       
        return image, target
    
    def filp_array(self, array, flipCode):
        if flipCode != -1:
            array = np.flip(array, flipCode)
        elif flipCode == -1:
            array = np.flipud(array)
            array = np.fliplr(array)
        return array

# Weighting Function for Semantic Segmentation
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)
	