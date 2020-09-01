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








# Transform
# transform = transforms.Compose([transforms.ToTensor()])
data_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.RandomCrop(255),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class seeds_dataset(Dataset):
    def __init__(self, txt_path='seeds_dataset_labels_file.csv', img_dir='seeds_dataset/images/'):
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
            X = self.transform(X)   ##   H x W x C to    C x H x W

            

        def flatten(t):
            t = t.reshape(1, -1)
            t = t.squeeze()
            return t
        # X = flatten(X)
        # print(X.size())
        
        

        return X, Y

print('Data loaded')
# ds = seeds_dataset()
# print(len(ds))
# print(ds[0])
