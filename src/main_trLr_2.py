from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/seed_googlenet/')



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),            ###               Done, vertival flip , random_rotate , brightness and contrast 
        transforms.RandomVerticalFlip(p=0.5), 
        # transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),
        transforms.ToTensor(),
        transforms.Normalize(mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),
        transforms.ToTensor(),
        transforms.Normalize(mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026])  ## mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ]),
}





# Hyperparameters 
batch_size =128
num_epochs = 150


## ____ Loading the dataset ____  ####

data_dir = 'seeds_dataset/data1/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes
print(class_names)
print(batch_size)
print(num_epochs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  Defining the Traning Function  
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (Validation)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Loss and Accuracy 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            ## write loss and acc to tensorboard 
            if phase == 'train':
            	writer.add_scalar('Tranning Loss 10**-3', (running_loss / 1000), epoch * len(dataloaders) + batch_idx)
            	writer.add_scalar('Tranning Accuracy ', epoch_acc, epoch * len(dataloaders) + batch_idx)
            	# print('')

            if phase == 'validation':
                writer.add_scalar('Validation Loss:', (running_loss/1000), epoch * len(dataloaders) + batch_idx)
                writer.add_scalar('Validation Acc :', epoch_acc, epoch * len(dataloaders) + batch_idx)
            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc*100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
## __ choose the model __ ##

# model_ft = models.resnet18(pretrained=True)
model_ft = models.googlenet(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 4.
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

data_iter = iter(dataloaders['train'])
images, labels = data_iter.next()

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.99)

# Decay LR by a factor of 0.1 every **** epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)


## _______ Trainning model  __________ ####

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs)


## ____ Confusion Matrix _____##

print('Confusion Matrix')
confusion_matrix = torch.zeros(4, 4)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['validation']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)


#########_____________           TRASH             _________##############



# writer.add_scalar('training loss 10**-4',
#                             running_loss / 1000,
#                             epoch * len(train_loader) + batch_idx)
#         		writer.add_scalar('accuracy',
#                             running_correct ,
#                             epoch * len(train_loader) + batch_idx)