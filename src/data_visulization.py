# Data Visualization 


# Loading the Data
from load_data import seeds_dataset, seeds_dataset_no_Transformation

from PIL import Image

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



batch_size = 128 

train_txt_path='data/train_data_file.csv' 
train_img_dir='data/train'
test_text_path='data/test_data_file.csv'
test_img_dir='data/validation'

train_dataset = seeds_dataset(train_txt_path,train_img_dir)
test_dataset = seeds_dataset(test_text_path,test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
validation_loader = DataLoader(test_dataset, batch_size=batch_size)




data_iter = iter(validation_loader)
images, labels = data_iter.next()

# print(images.shape)
images = images.numpy()
# print(images.shape)

f, axarr = plt.subplots(8,8)
img_no = 0
for i in range(8):
	for j in range(8):
	
		image = images[img_no, :, :, :]
		image = np.moveaxis(image, 0, -1)  #  3 x 128 x 128 To 128 x 128 x 3
		axarr[i,j].imshow(image)
		img_no += 1
		
plt.show()






#######    TRASH     ##############
#######	   TRASH	 ##############



# image.save('my_seed1.png')

# print(image.shape)
# image = np.moveaxis(image, 0, -1)  #  3 x 128 x 128 To 128 x 128 x 3
# # # print(image)
# plt.imshow(image, interpolation='nearest')
# print(image.shape)
# imgplot = plt.imshow(image)

# image = np.reshape(image, (128, 128))



# img = Image.fromarray(image, 'RGB')
# img.save('my.png')
# img.show()