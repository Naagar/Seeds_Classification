from __future__ import print_function, division

import torch

import argparse



import random
from PIL import Image

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

from tensorboardX import SummaryWriter
# Hyperparameters 
# batch_size = 256
# num_epochs = 70

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/resnet18_new_45k/')

parser = argparse.ArgumentParser(description='Cutmix PyTorch seeds Training')

parser.add_argument('--num_epochs', default=70, type=int,
                    help='No. of Epochs')
parser.add_argument('--batch_size', default=512, type=int,
                    help='Batch size')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
args = parser.parse_args()


# Cut-MIX   ############
######### #########

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
#  -------------- ### ------------------ #
# Salt and Pepper noice

class AddPepperNoise(object):
    """Increase salt and pepper noise
    Args:
        snr （float）: Signal Noise Rate
                 p (float): probability value, perform the operation according to probability
    """
    # The default signal-to-noise ratio is 90%, and 90% of the pixels are the original image
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # Set the percentage of the signal SNR
            signal_pct = self.snr
            # Percentage of noise
            noise_pct = (1 - self.snr)
            # Select the mask mask value 0, 1, 2 0 represents the original image 1 represents salt noise 2 represents pepper noise
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # Salt noise white
            img_[mask == 2] = 0     # Pepper noise black
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),
        AddPepperNoise(0.9, p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),            ###               Done, vertival flip , random_rotate , brightness and contrast 
        transforms.RandomVerticalFlip(p=0.5), 
        # transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),
        transforms.ToTensor(),
        transforms.Normalize(mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        # transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),
        transforms.ToTensor(),
        transforms.Normalize(mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026])  ## mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ]),
}




## ____ Loading the dataset ____  ####

data_dir = 'seeds_dataset/data/' # data1 #data_plus_fake
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validation']}
# print class names and hyperparameters
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes
print('class names:',class_names)
print('Batch size:',args.batch_size)
print('No. of Epochs:',args.num_epochs)


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
                    r = np.random.rand(1)
                    if args.beta > 0 and r < args.cutmix_prob:
                        # generate mixed sample
                        lam = np.random.beta(args.beta, args.beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        # compute output
                        outputs = model(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                    else:
                        # compute output
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)


                    # outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # loss = criterion(outputs, labels)

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
                writer.add_scalar('Validation Loss 10**-3:', (running_loss / 1000), epoch * len(dataloaders) + batch_idx)
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

# model_ft = models.alexnet(pretrained=True)
# model_ft = models.squeezenet1_0(pretrained=True)
# model_ft = models.vgg16(pretrained=True)
# model_ft = models.densenet161(pretrained=True)
# model_ft = models.inception_v3(pretrained=True)
# model_ft = models.googlenet(pretrained=True)
# model_ft = models.shufflenet_v2_x1_0(pretrained=True)
# model_ft = models.mobilenet_v2(pretrained=True)
# model_ft = models.resnext50_32x4d(pretrained=True)
# model_ft = models.wide_resnet50_2(pretrained=True)
# model_ft = models.mnasnet1_0(pretrained=True)
model_ft = models.resnet18(pretrained=True)
# model_ft = models.googlenet(pretrained=True)


# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 4.

# model_ft.fc = nn.Linear(num_ftrs, 4)



## Resume 
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

if torch.cuda.device_count() > 1:
  print("Using ", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

data_iter = iter(dataloaders['train'])
images, labels = data_iter.next()

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.99)

# Decay LR by a factor of 0.1 every **** epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)   ## set the step size   


## _______ Trainning model  __________ ####

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       args.num_epochs)


## ____ Confusion Matrix _____##

print('Confusion Matrix: 1458 1511 1419 1451')
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


########### --------------------------------         ##############


# list of image names with confusion 

import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# EXAMPLE USAGE:
# instantiate the dataset and dataloader
data_dir = 'seeds_dataset/data/validation/'
dataset = ImageFolderWithPaths(data_dir, data_transforms['validation']) # our custom dataset
dataloader = torch.utils.data.DataLoader(dataset)

print('Classwise accuracy :')

num_correct = 0
num_samples = 0
n_class_correct = [0 for i in range(4)]
n_class_samples = [0 for i in range(4)]
# actual_y = torch.tensor([])
# pred_y = torch.tensor([])
model_ft.eval()
classes = ('Broken', 'Discolored', 'Pure', 'Silkcut')
with torch.no_grad():

    for i, (x, y) in enumerate(dataloaders['validation']):

        x = x.to(device=device)
        y = y.to(device=device)

        # classes = ('Discolored', 'Pure', 'Broken', 'Silkcut')
        # classes = classes.to(device=device)


        scores = model_ft(x)

        _, predictions = scores.max(1)
        # Getting the confusing labels 
        # print(y)
        # print(predictions)
        # print(path)
        # print(a)
        num_correct += (predictions == y ).sum()
        num_samples += predictions.size(0)
            
        
        if ( args.batch_size == len(predictions) ):
            # for confusion matrix 
            # actual_y += y 
            # pred_y += predictions
            # for accuracy of each class 
            for i in range(args.batch_size):

                label = y[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1 

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    # classes = ('Discolored', 'Pure', 'Broken', 'Silkcut')
    # classes = classes.to(device=device)
    for i in range(4):
        acc = 100.0 *n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}, {acc} %')






#iterate over val_data to check the label miss match

with torch.no_grad():

    for  x, y, (path) in dataloader:

        x = x.to(device=device)
        y = y.to(device=device)

        # classes = ('Discolored', 'Pure', 'Broken', 'Silkcut')
        # classes = classes.to(device=device)


        scores = model_ft(x)

        _, predictions = scores.max(1)
        # Getting the confusing labels 
        print(y)
        print(predictions)
        print(path)
        # print(a)
        num_correct += (predictions == y ).sum()
        num_samples += predictions.size(0)
            
        
        if ( args.batch_size == len(predictions) ):
            # for confusion matrix 
            # actual_y += y 
            # pred_y += predictions
            # for accuracy of each class 
            for i in range(args.batch_size):

                label = y[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1 

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    # classes = ('Discolored', 'Pure', 'Broken', 'Silkcut')
    # classes = classes.to(device=device)
    # for i in range(4):
    #     acc = 100.0 *n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {classes[i]}, {acc} %')



#########_____________           TRASH             _________##############



# writer.add_scalar('training loss 10**-4',
#                             running_loss / 1000,
#                             epoch * len(train_loader) + batch_idx)
#         		writer.add_scalar('accuracy',
#                             running_correct ,
#                             epoch * len(train_loader) + batch_idx)



#### models


# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)