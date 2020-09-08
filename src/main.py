## seeds classification
from __future__ import print_function, division
import os

from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch 
import torch.nn as nn
import torchvision.transforms as transforms 
import torch.optim as optim  ##, lr_scheduler
import torchvision 
import torch.nn.functional as F

# sampler 
from torch.utils.data.sampler import SubsetRandomSampler

## to load dataset 
from load_data import seeds_dataset                           # load the data Set

##    model load
from model import seeds_model
# from resnet_from_scratch import ResNet, block
from resnet_18_34 import ResNet, BasicBlock
from MobileNet import MobileNetV2
## Symmery Writer to visualize the training loss
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/seed_MobileNet_e_140_img_128_val_30/')

# ResNet18 
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# ResNet34
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50(img_channel=3, num_classes=4):

    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=4):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=4):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# def test():
#     net = ResNet50(img_channel=3, num_classes=4)
#     y = net(torch.randn(4, 3, 224, 224))
#     print(y.size())
# test()



#set device 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# hyper parameters 

in_channels = 3
num_classes = 4

learning_rate = 0.0001      ##  default  1e-3
batch_size = 16             ##  default  256  for best data augmentation
num_epochs = 140            ##  default  100

validation_split = .3       ##  20% 
shuffle_dataset = True 
random_seed= 42



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


transformed_dataset = seeds_dataset()

# Creating data indices for training and validation splits:
dataset_size = len(transformed_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(transformed_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = DataLoader(transformed_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)



# train_loader = DataLoader(transformed_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=0)


# model  = seeds_model()
model = MobileNetV2(width_mult=1)
# model = ResNet18()

# model = ResNet50(img_channel=3, num_classes=4)
# model = ResNet101(img_channel=3,num_classes=4)
# model = ResNet152(img_channel=3, num_classes=4)


# FILE = "model.pth"
# torch.save(model, FILE)

data_iter = iter(train_loader)
images, labels = data_iter.next()


grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)


model.to(device)

# Loss and optimizer
# print(train_loader[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.1)

# writer.add_graph(model, images)

# Train Network
running_loss = 0.0 
running_correct = 0.0

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

        # Visualization with Tensorboard 
        # scheduler.Step
        running_loss += loss.item()
        _, predictions = torch.max(scores.data, 1) 
        running_correct += (predictions == targets).sum().item()

         # ...log the running loss
        writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_loader) + batch_idx)
        writer.add_scalar('accuracy',
                            running_correct ,
                            epoch * len(train_loader) + batch_idx)

        # writer.close()
        
        running_loss = 0.0
        running_correct = 0.0
    # writer.add_scalar('Loss/train', losses, epoch)
    
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


print('Checking accuracy on Validation Set')

check_accuracy(validation_loader, model )

print('in_channels: ', in_channels)
print('batch_size: ', batch_size)
print('num_epochs: ', num_epochs)
print('learning_rate: ', learning_rate)
print('model: ', "MobileNet")
print('img_size:' '3 x 128 x 128')
print('Validation Split:', validation_split)
print('device:', device)

# torch.save(arg, path)
# torch.load(path)
# model.load_state_dict(arg)

# torch.save(model.state_dict(), PATH)


# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()