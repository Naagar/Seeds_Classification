## seeds classification
from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch 
import torch.nn as nn
import torchvision.transforms as transforms 
import torch.optim as optim
import torchvision 

## to load dataset 
from load_data import seeds_dataset                           # load the data Set
         ##    model 
from model import seeds_model

#set device 
 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# hyper parameters 

in_channels = 3
num_classes = 4

learning_rate = 1e-3
batch_size = 64             ##  default  256 
num_epochs = 100



# datalaoder  load data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


transformed_dataset = seeds_dataset()




train_loader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


model  = seeds_model()

model.to(device)

# Loss and optimizer
# print(train_loader[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network 

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)


        # Forward 

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())


        # Backward 

        optimizer.zero_grad()
        loss.backward()


        # Gradient Descent or adam step 
        optimizer.step()

    print(f'cost at each epoch {epoch} is {sum(losses)/len(losses)} ')

# checking accurscy on training set

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()


    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)


            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y ).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accurscy {float(num_correct)/float(num_samples)*100}')
    model.train()
print('Checking accuracy on Traning set')

check_accuracy(train_loader, model)


# print('Checking accuracy on Test Set')

# check_accuracy(train_loader, model )



















# # read the CSV containing the images names and labels/classes
    
# annotations = pd.read_csv('seeds_dataset_labels_file.csv') 
# root_dir = 'seeds_dataset/images'


# n = 1
# img_name = annotations.iloc[n, 0]
# label = annotations.iloc[n, 1]
# label = np.asarray(label)




# seed_dataset = Seeds_Dataset(csv_file='seeds_dataset_labels_file.csv',
#                                     root_dir='seeds_dataset/images/')




# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# data_set = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)

# dataset = seeds_dataset(csv_file = 'seeds_dataset_labels_file.csv', root_dir = 'seeds_dataset/images',
#                       transform = transforms.ToTensor())



# train_set, test_set = random_split(dataloader, [.8, .2])

# train_loader = DataLoader(dataset=dataloader, batch_size = batch_size, shuffle=True, num_workers=0 )

# class Seeds_Dataset(Dataset):
#     """Seeds dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.annotations.iloc[idx, 0])
#         image = io.imread(img_name)
#         label = self.annotations.iloc[idx, 1:]
#         label = np.array([label])
#         print('labels:', label)
#         label = label.astype('float')
#         print(label)
#         sample = {'image': image, 'label': label}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample



























































# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torchvision import transforms
# from torch.utils.data.dataset import Dataset  # For custom datasets


# # in-repo imports
# from load_data import seeds_dataset

# # from custom_dataset_from_csv import DatasetFromCsvLocation, Rescale, RandomCrop, ToTensor

# from model import seeds_model



# no_epochs = 4

# if __name__ == "__main__":
    
#     # Dataset variant :
#     # Read image locations from Csv - transformations are defined inside dataset class
#     data_from_csv =  seeds_dataset()


#     seed_dataset_loader = torch.utils.data.DataLoader(dataset=data_from_csv,
#                                                     batch_size=4,
#                                                     shuffle=True, pin_memory=True)
#     model = seeds_model()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#     for epoch in range(no_epochs):  # loop over the dataset multiple times

#         running_loss = 0.0

#         for i, data in enumerate(seed_dataset_loader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0

#     print('Finished Training')

