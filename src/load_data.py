from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random

import io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch
from torchvision import transforms


lambd = 1
# Transform
data_transforms = transforms.Compose([
        transforms.Resize(128),                            ## 128
        transforms.RandomRotation(25),                     ## 256
        transforms.RandomCrop(128),                        ## 256
        # transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),            ###               Done, vertival flip , random_rotate , brightness and contrast 
        transforms.RandomVerticalFlip(p=0.5),              ##
        # transforms.RandomRotate(),
        # transforms.Lambda(lambd)
        
        transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),   ##   new,  brightness=0.19,saturation=0.5,contrast=0.12, hue=0.1, 
        # (5)brightness=0.15,saturation=0.10,contrast=0.15, hue=0.1
        # (4)brightness=0.10,saturation=0.90,contrast=0.9, hue=0.9
        # (3)brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09
        # (2)brightness=0.30,saturation=0.290,contrast=0.19, hue=0.19
        # (1)brightness=0.40,saturation=0.190,contrast=0.159, hue=0.39
        #  transforms.ColorJitter(brightness=0.19,saturation=0.5,contrast=0.02, hue=0.1),   ##   new 
        # 
        transforms.ToTensor(),
        # transforms.functional.adjust_gamma(img, gamma, gain=1)
        transforms.Normalize([1.0817, 1.0146, 0.8792], [0.8482, 0.8573, 1.1026])  #   mean : [1.0817, 1.1146, 0.9792], std :[0.8482, 0.9573, 1.1026] 
    ])


class seeds_dataset(Dataset):
    def __init__(self, txt_path, img_dir):

        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """


        self.data_info = pd.read_csv(txt_path, header=None)
        
        self.labels = np.asarray(self.data_info.iloc[:, 1])

        df = pd.read_csv(txt_path, sep=' ', index_col=0)          ## changed part -1 
        self.data_len = len(self.data_info.index)
        # self.img_names = df.index.values
        self.img_names = np.asarray(self.data_info.iloc[:, 0]) 
        # self.img_names = 'new_' + self.img_names


        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = data_transforms
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.get_image_selector = True if img_dir.__contains__('tar') else False
        self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None

    
    def get_image_from_folder(self, name):
        

        image = Image.open(os.path.join(self.img_dir, name))
        return image

    def __len__(self):
        
        return len(self.img_names)

    def __getitem__(self, index):
        

        if index == (self.__len__() - 1) and self.get_image_selector:  # close tarfile opened in __init__
            self.tf.close()

        if self.get_image_selector:  # note: we prefer to extract then process!
            X = self.get_image_from_tar(self.img_names[index])
        else:
            X = self.get_image_from_folder(self.img_names[index])

    

        
        # Y = self.labels
        Y = self.labels[index]
        # print(Y)
        if self.transform is not None:
            X = self.transform(X)   ##   H x W x C to    C x H x W and some other transformations 
        
        

        return X, Y






### TRASH 

# class seeds_dataset_test(Dataset):
#     def __init__(self, txt_path='seeds_dataset/data/test_data_file.csv', img_dir='seeds_dataset/data/validation'):
#         """
#         Initialize data set as a list of IDs corresponding to each item of data set

#         :param img_dir: path to image files as a uncompressed tar archive
#         :param txt_path: a text file containing names of all of images line by line
#         :param transform: apply some transforms like cropping, rotating, etc on input image
#         """


#         self.data_info = pd.read_csv(txt_path, header=None)
        
#         self.labels = np.asarray(self.data_info.iloc[:, 1])

#         df = pd.read_csv(txt_path, sep=' ', index_col=0)          ## changed part -1 
#         self.data_len = len(self.data_info.index)
#         # self.img_names = df.index.values
#         self.img_names = np.asarray(self.data_info.iloc[:, 0]) 
#         # self.img_names = 'new_' + self.img_names


#         self.txt_path = txt_path
#         self.img_dir = img_dir
#         self.transform = data_transforms
#         self.to_tensor = ToTensor()
#         self.to_pil = ToPILImage()
#         self.get_image_selector = True if img_dir.__contains__('tar') else False
#         self.tf = tarfile.open(self.img_dir) if self.get_image_selector else None

    
#     def get_image_from_folder(self, name):
        

#         image = Image.open(os.path.join(self.img_dir, name))
#         return image

#     def __len__(self):
        
#         return len(self.img_names)

#     def __getitem__(self, index):
        

#         if index == (self.__len__() - 1) and self.get_image_selector:  # close tarfile opened in __init__
#             self.tf.close()

#         if self.get_image_selector:  # note: we prefer to extract then process!
#             X = self.get_image_from_tar(self.img_names[index])
#         else:
#             X = self.get_image_from_folder(self.img_names[index])

    

        
#         # Y = self.labels
#         Y = self.labels[index]
#         # print(Y)
#         if self.transform is not None:
#             X = self.transform(X)   ##   H x W x C to    C x H x W and some other transformations 

            

#         def flatten(t):
#             t = t.reshape(1, -1)
#             t = t.squeeze()
#             return t
#         # X = flatten(X)
#         # print(X.size())
        
        

#         return X, Y


# print('Data loaded')
# ds = seeds_dataset()
# print(len(ds))
# print(ds[0])
    # def __init__(self, txt_path='seeds_dataset/data/test_data_file.csv', img_dir='seeds_dataset/data/validation'):
