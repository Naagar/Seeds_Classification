#  loading Images

# %matplotlib inline
# %config InlineBackend.figure_format = ‘retina’
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
# import helper


# data_dir = 'images'
# transform = transforms.Compose([transforms.Resize(255),
# 								transforms.CenterCrop(224),
# 								transforms.ToTensor()])

# dataset = datasets.ImageFolder(data_dir, transform=transform)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# images, labels = next(iter(dataloader))
# # helper.imshow(images[0], normalize=False)
# print(images[1])



# dataset = datasets.ImageFolder('/images', transform=transform)


# transform = transforms.Compose([transforms.Resize(255),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor()])


# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# images, labels = next(iter(dataloader))
# helper.imshow(images[0], normalize=False)
'''


motion captu

'''

# ## Introducing Randomness
# train_transforms = transforms.Compose([
#                                 transforms.RandomRotation(30),
#                                 transforms.RandomResizedCrop(224),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor()])


# # Norimalize images

# input[channel] = (input[channel] - mean[channel]) / std[channel]



data_dir = 'images'

#Applying Transformation
data_transforms = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

# test_transforms = transforms.Compose([transforms.Resize(255),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor()])
data = datasets.ImageFolder(data_dir,  
                                    transform=train_transforms)                                       
# test_data = datasets.ImageFolder(data_dir + ‘/test’, 
#                                     transform=test_transforms)

#Data Loading
data_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=32)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


