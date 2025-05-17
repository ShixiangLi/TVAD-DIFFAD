import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random
from data.perlin import rand_perlin_2d_np

texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
class MVTecTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class MVTecTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):

        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        

        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground_path(self):
        foreground_path_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_path_id]
        return foreground_path


    def get_foreground_mvtec(self,image_path):
        classname = self.classname
        if classname in texture_list:
            foreground_path = self.random_choice_foreground_path()
        else:
            foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path



    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  
                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)  

                msk = (object_perlin).astype(np.float32) 
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
                
            
            if self.classname in texture_list: # only DTD
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else: # DTD and self-augmentation
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  # >0.5 is DTD 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else: #self-augmentation
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground_mvtec(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0 
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample



class VisATestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.JPG"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class VisATrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        self.image_paths = sorted(glob.glob(self.root_dir+"/*.JPG"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    
    def __len__(self):
        return len(self.image_paths)


    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  

            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class DAGMTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.PNG"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_label.PNG"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class DAGMTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.PNG"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground(self):
        foreground_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_id]
        return foreground_path

    def get_foreground(self,image_path):
        foreground_path = self.random_choice_foreground()
        return foreground_path 

    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6
            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0:
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)

                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  
        image = np.array(image).astype(np.float32)/255.0

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class MPDDTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class MPDDTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        

    
    def __len__(self):
        return len(self.image_paths)

    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5: 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate((np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  # [0,255]->[0,1]
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class CustomTestDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args=None): # Added args for consistency, though not strictly used here yet
        self.root_dir = os.path.join(data_path, 'test') # Assumes data_path is 'datasets/your_custom_dataset_root'
        self.classname = classname # User might want to specify sub-folders if their custom dataset has them
        self.img_size = img_size # Expecting [64, 64]
        
        # Adjust glob pattern if your images have a different extension (e.g., *.jpg)
        # This example assumes a structure like:
        # your_custom_dataset_root/test/good/image1.png
        # your_custom_dataset_root/test/anomaly_type/image2.png
        # your_custom_dataset_root/ground_truth/anomaly_type/image2_mask.png (or similar)
        self.images = []
        good_images = sorted(glob.glob(os.path.join(self.root_dir, 'good', "*.png")))
        self.images.extend(good_images)
        
        # Find anomaly folders. You might need to adjust this logic based on your exact folder structure.
        # This example assumes all subdirectories in 'test' other than 'good' are anomaly types.
        potential_anomaly_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and d != "good"]
        for anomaly_type in potential_anomaly_dirs:
            anomaly_images = sorted(glob.glob(os.path.join(self.root_dir, anomaly_type, "*.png")))
            self.images.extend(anomaly_images)
            
        if not self.images:
            print(f"Warning: No images found in {self.root_dir} for class {self.classname}. Check dataset path and structure.")

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupt: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Mask not found for {image_path}, using empty mask: {mask_path}")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize
        image = cv2.resize(image, dsize=(self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, dsize=(self.img_size[1], self.img_size[0]))

        # Normalize
        image = image / 255.0
        mask = mask / 255.0 # Ensure mask is 0 or 1

        # Reshape and type conversion
        image = np.array(image, dtype=np.float32) # Shape (H, W, C)
        mask = np.array(mask, dtype=np.float32)   # Shape (H, W)
        mask = np.expand_dims(mask, axis=2)       # Shape (H, W, 1)

        # Transpose
        image = np.transpose(image, (2, 0, 1)) # (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))   # (1, H, W)
        
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        # Determine if the image is 'good' or an anomaly based on its path
        # Example: datasets/your_custom_dataset_root/test/good/image.png
        # Example: datasets/your_custom_dataset_root/test/defect_type/image.png
        
        parts = img_path.split(os.sep)
        # parts[-2] should be 'good' or the anomaly type folder name.
        # parts[-3] should be 'test'.
        # parts[-4] should be the classname (if you have sub-classes, otherwise it's the root dataset name)
        
        image_type_folder = parts[-2] 
        file_name = parts[-1]

        if image_type_folder == 'good':
            mask_path = None
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            # Construct mask path - this assumes a parallel 'ground_truth' directory
            # Example: datasets/your_custom_dataset_root/ground_truth/test/defect_type/image_mask.png
            # You MUST adjust this logic to match your ground truth mask storage.
            # A common MVTec-like structure is ../../ground_truth/defect_type/image_mask.png from the image file.
            # Let's assume a structure relative to the data_path provided in __init__
            # data_path = "datasets/your_custom_dataset_root"
            # Mask path example: datasets/your_custom_dataset_root/ground_truth/test/anomaly_type_folder/mask_file.png
            mask_file_name = file_name.split('.')[0] + ".png" # Assuming mask names
            mask_dir = os.path.join(os.path.dirname(self.root_dir).replace('test', 'ground_truth'), image_type_folder)
            # A simpler assumption if ground_truth is parallel to your_custom_dataset_root:
            # custom_root_dir = os.path.dirname(os.path.dirname(self.root_dir)) # e.g. datasets/your_custom_dataset_root
            # mask_path = os.path.join(custom_root_dir, 'ground_truth', image_type_folder, mask_file_name)
            # For MVTec-like structure:
            base_truth_dir = os.path.abspath(os.path.join(self.root_dir, "..", "ground_truth")) # Path relative to data_path/classname
            if not os.path.exists(base_truth_dir): # Fallback if ground_truth is not two levels up from classname/test/type
                 base_truth_dir = os.path.join(os.path.dirname(os.path.dirname(self.root_dir)), 'ground_truth', self.classname) # Path relative to data_path

            mask_path = os.path.join(base_truth_dir, image_type_folder, mask_file_name)
            if not os.path.exists(mask_path):
                # Fallback for structure like VisA: ground_truth/image_type_folder/mask_file.png (relative to test folder)
                mask_path_alt = os.path.join(os.path.dirname(img_path), '../../ground_truth/', image_type_folder, file_name.replace(".png", "_mask.png")) # More MVTec like
                if os.path.exists(mask_path_alt):
                    mask_path = mask_path_alt
                else: # Final fallback, assuming mask is in same folder with _mask suffix
                    mask_path_same_folder = img_path.replace(".png", "_mask.png")
                    if os.path.exists(mask_path_same_folder):
                        mask_path = mask_path_same_folder
                    else:
                        print(f"Warning: Mask file not found at {mask_path} or {mask_path_alt} or {mask_path_same_folder}. Using empty mask for {img_path}.")
                        mask_path = None # Will create an empty mask in transform_image

            has_anomaly = np.array([1], dtype=np.float32)
        image, mask = self.transform_image(img_path, mask_path)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx, 'file_name': os.path.join(image_type_folder, file_name)}
        return sample
    
class CustomTrainDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args):
        self.classname = classname # User might want to specify sub-folders
        self.root_dir = os.path.join(data_path, 'train', 'good') # Assumes data_path is 'datasets/your_custom_dataset_root'
        self.img_size = img_size # Expecting [64, 64]
        self.anomaly_source_path = args["anomaly_source_path"] # Path to DTD or similar

        # Adjust glob pattern if your images have a different extension
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "*.png")))
        if not self.image_paths:
            print(f"Warning: No training images found in {self.root_dir}. Check dataset path and structure.")

        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "images", "*", "*.jpg")))
        if not self.anomaly_source_paths:
            print(f"Warning: No anomaly source images (DTD) found in {self.anomaly_source_path}. Synthetic anomalies might be limited.")

        # Augmenters (can be kept as they are generally useful)
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        if not self.image_paths: # Prevent error if no images found
            return 0
        return len(self.image_paths)


    def randAugmenter(self): # Used for augmenting the anomaly source or the image for self-augmentation
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        return aug

    def perlin_synthetic(self, image, cv2_image_orig, anomaly_source_path):
        # image: numpy array, normalized (0-1), shape (H, W, C)
        # cv2_image_orig: numpy array, original (0-255), shape (H, W, C)
        
        # 50% chance to return the image without anomaly
        if random.random() > 0.5:
            return image.astype(np.float32), \
                   np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        perlin_scale = 6
        min_perlin_scale = 0
        
        has_anomaly_mask = 0
        try_count = 0
        target_msk = None

        while has_anomaly_mask == 0 and try_count < 20: # Try a few times to get a valid mask
            perlin_scalex = 2 ** (random.randint(min_perlin_scale, perlin_scale -1)) # Adjust range if needed for 64x64
            perlin_scaley = 2 ** (random.randint(min_perlin_scale, perlin_scale -1))

            perlin_noise = rand_perlin_2d_np(
                (self.img_size[0], self.img_size[1]), (perlin_scalex, perlin_scaley)
            )
            perlin_noise = self.rot(image=perlin_noise) # Rotate the Perlin noise mask

            threshold = 0.5 # Threshold for Perlin noise to create binary mask
            perlin_binary_mask = np.where(perlin_noise > threshold, 1.0, 0.0)
            
            # Ensure the mask is not too small or too large (optional, but good practice)
            mask_sum = np.sum(perlin_binary_mask)
            if mask_sum > (0.05 * self.img_size[0] * self.img_size[1]) and \
               mask_sum < (0.6 * self.img_size[0] * self.img_size[1]): # Min 5%, Max 60% coverage
                has_anomaly_mask = 1
                target_msk = perlin_binary_mask
            try_count += 1
        
        if target_msk is None: # If no suitable mask found after retries, return original
             return image.astype(np.float32), \
                   np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        target_msk = np.expand_dims(target_msk, axis=2).astype(np.float32) # Shape (H, W, 1)

        # Decide anomaly content: DTD or self-augmentation
        # For simplicity, let's try to use DTD if available, otherwise self-augmentation.
        # Or, make it a 50/50 choice if DTD is available.
        
        use_dtd_anomaly = True if self.anomaly_source_paths and random.random() > 0.5 else False
        
        img_object_thr = np.zeros_like(image, dtype=np.float32)

        if use_dtd_anomaly:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            if anomaly_source_img is None: # Fallback if DTD image fails to load
                use_dtd_anomaly = False 
            else:
                anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.img_size[1], self.img_size[0]))
                
                aug = self.randAugmenter()
                anomaly_img_augmented = aug(image=anomaly_source_img) # Augment DTD image
                img_object_thr = anomaly_img_augmented.astype(np.float32) / 255.0 * target_msk
        
        if not use_dtd_anomaly: # Self-augmentation (e.g., CutPaste like)
            aug = self.randAugmenter()
            # Use the original unnormalized image for augmentation, then normalize
            anomaly_patch_source = aug(image=cv2_image_orig.copy()) 
            
            # Simple CutPaste: take a random patch and place it
            # More sophisticated methods can be implemented here
            # For now, let's use a strongly augmented version of the original image
            img_object_thr = anomaly_patch_source.astype(np.float32) / 255.0 * target_msk

        beta = random.uniform(0.2, 0.8) # Blending factor

        # image is already normalized (0-1)
        augmented_image = image * (1 - target_msk) + \
                          (1 - beta) * img_object_thr + \
                          beta * image * target_msk

        augmented_image = np.clip(augmented_image, 0.0, 1.0).astype(np.float32)
        
        return augmented_image, target_msk, np.array([1.0], dtype=np.float32)


    def __getitem__(self, idx):
        # In training, we always pick a random image if image_paths is not empty
        if not self.image_paths:
             # Return a dummy sample if no images are available to prevent crashes
            dummy_image = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)
            dummy_mask = np.zeros((1, self.img_size[0], self.img_size[1]), dtype=np.float32)
            return {'image': dummy_image, "anomaly_mask": dummy_mask,
                    'augmented_image': dummy_image, 'has_anomaly': np.array([0.0], dtype=np.float32), 
                    'idx': -1}

        actual_idx = random.randint(0, len(self.image_paths) - 1)
        image_path = self.image_paths[actual_idx]
        
        cv2_image_orig = cv2.imread(image_path) # For self-augmentation, keep original BGR 0-255
        if cv2_image_orig is None:
            raise FileNotFoundError(f"Training image not found or corrupt: {image_path}")
        
        # Prepare normalized image for background
        image_rgb = cv2.cvtColor(cv2_image_orig, cv2.COLOR_BGR2RGB)
        image_rgb_resized = cv2.resize(image_rgb, dsize=(self.img_size[1], self.img_size[0]))
        image_normalized = image_rgb_resized.astype(np.float32) / 255.0 # Shape (H, W, C)

        # Resize original cv2_image for self-augmentation source
        cv2_image_orig_resized = cv2.resize(cv2_image_orig, dsize=(self.img_size[1], self.img_size[0]))


        anomaly_source_idx = 0
        anomaly_path = None
        if self.anomaly_source_paths: # Check if DTD paths are available
            anomaly_source_idx = random.randint(0, len(self.anomaly_source_paths) - 1)
            anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        
        augmented_image_np, anomaly_mask_np, has_anomaly_np = self.perlin_synthetic(
            image_normalized, cv2_image_orig_resized, anomaly_path
        )
        
        # Transpose for PyTorch (C, H, W)
        image_final = np.transpose(image_normalized, (2, 0, 1)) # Original normal image
        augmented_image_final = np.transpose(augmented_image_np, (2, 0, 1))
        anomaly_mask_final = np.transpose(anomaly_mask_np, (2, 0, 1))


        sample = {'image': image_final, 
                  "anomaly_mask": anomaly_mask_final,
                  'augmented_image': augmented_image_final, 
                  'has_anomaly': has_anomaly_np, 
                  'idx': actual_idx}

        return sample