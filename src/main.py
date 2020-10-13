#  test and train, each class % in test     Done
# bad distribution bw test and train        Done
# random_mini batch                         Done

# first verify the augmantanion             Done 

# @geevi github                             Done

# record everything,                        Done
# readme Done
# organization of the code                  Done 
#02/08
## lr_scheduling                            Done
## Visualization                            Done

## Look at the traing curve                 Done

## 

## seeds classification
from __future__ import print_function, division
import os
import warnings
warnings.filterwarnings("ignore")

from skimage import io, transform
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from plotcm import plot_confusion_matrix
# from resources.plotcm import plot_confusion_matrix

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
from load_data import seeds_dataset                          # load the data Set

##    model load
import torchvision.models as models

from model import seeds_model
# from squeezenet_11 import SqueezeNet
from resnet_from_scratch import ResNet, block
# from resnet_18_34 import ResNet, BasicBlock
from MobileNet import MobileNetV2
## Symmery Writer to visualize the training loss
from torch.utils.tensorboard import SummaryWriter

# LR shcedular

from torch.optim.lr_scheduler import ReduceLROnPlateau


# Ignore warnings

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/seed_resnet18_e_120/')

# hyper parameters 
in_channels = 3
num_classes = 4

learning_rate = 0.01       ##  default  1e-3
batch_size = 128             ##  default  256  for best data augmentation
num_epochs = 200            ##  default  100

# validation_split = .3     ##  20% 
shuffle_dataset = True 

# classes names 
classes = ('Discolored', 'Pure', 'Broken', 'Silkcut') 

global loss

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





#set device 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# dataset paths 
train_txt_path='seeds_dataset/data/train_data_file.csv' 
train_img_dir='seeds_dataset/data/train'
test_text_path='seeds_dataset/data/test_data_file.csv'
test_img_dir='seeds_dataset/data/validation'

# Load dataset. train and test
train_dataset = seeds_dataset(train_txt_path,train_img_dir)
test_dataset = seeds_dataset(test_text_path,test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
validation_loader = DataLoader(test_dataset, batch_size=batch_size)


### Selecting model for training 

vgg16 = models.vgg16()
resnet18 = models.resnet18()
resnet34 = models.resnet34()
squeezenet = models.squeezenet1_0()

# model = vgg16
model = resnet18
# model = resnet34
# model = squeezenet
# model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True) ## Pretrained 
# model = SqueezeNet()
# model  = seeds_model()
# model = MobileNetV2(width_mult=1)
# model = ResNet18()

# model = ResNet50(img_channel=3, num_classes=4)
# model = ResNet101(img_channel=3,num_classes=4)
# model = ResNet152(img_channel=3, num_classes=4)


data_iter = iter(train_loader)
images, labels = data_iter.next()

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  ## momentum , weight_decay=weight_decay
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.98, nesterov=True)  # momentum=0.19, 0.89, .92, 0.98

# optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=5000.0, weight_decay=0)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.10)  # step_size=70, 80, 60
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=False)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.31)

### if 

# optimizer_test = optim.SGD([
#                 {'params': model.base.parameters()},
#                 {'params': model.classifier.parameters(), 'lr': 1e-3}
#             ], lr=1e-2, momentum=0.9)


## Saving the best model 
PATH = 'best_model/'

# Loading the model to cuda device
model.to(device)


# Training  Network
running_loss = 0.0 
running_correct = 0.0

best_accuracy = 0


for epoch in range(num_epochs):
    losses = []
    loss = 0
    # LR scheduling
    # print('Epoch:', epoch,'LR:', scheduler.get_lr())
    scheduler.step()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)


        # Forward 
        # global loss 
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())


        # Backward 

        optimizer.zero_grad()
        loss.backward()

        # Clipping 
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)


        # Gradient Descent or adam step 
        optimizer.step()

        # Visualization with Tensorboard 
        # scheduler.step()
        running_loss += loss.item()
        _, predictions = torch.max(scores.data, 1) 
        running_correct += (predictions == targets).sum().item()

         # ...log the running loss
        writer.add_scalar('training loss 10**-4',
                            running_loss / 1000,
                            epoch * len(train_loader) + batch_idx)
        writer.add_scalar('accuracy',
                            running_correct ,
                            epoch * len(train_loader) + batch_idx)

        # writer.close()
        cur_accuracy = running_correct
        if cur_accuracy > best_accuracy:
                torch.save(model.state_dict(), 'best_model/best_model.pt')
                best_accuracy = cur_accuracy
        
        running_loss = 0.0
        running_correct = 0.0

    # writer.add_scalar('Loss/train', losses, epoch)
    # scheduler.step(loss)
    print(f'cost at each epoch {epoch} is {sum(losses)/len(losses):.4f}')

# checking accurscy on training set

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    # actual_y = torch.tensor([])
    # pred_y = torch.tensor([])
    model.eval()


    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)


            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y ).sum()
            num_samples += predictions.size(0)
                
            
            if ( batch_size == len(predictions) ):
                # for confusion matrix 
                # actual_y += y 
                # pred_y += predictions
                # for accuracy of each class 
                for i in range(batch_size):

                    label = y[i]
                    pred = predictions[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1 

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
        
        for i in range(4):
            acc = 100.0 *n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}, {acc} %') 

    model.train()



print('Checking accuracy on Traning set')

check_accuracy(train_loader, model)


print('Checking accuracy on Validation Set')

check_accuracy(validation_loader, model )

print('in_channels: ', in_channels)
print('batch_size: ', batch_size)
print('num_epochs: ', num_epochs)
print('learning_rate: ', learning_rate)
print('model: ', "model")
print('img_size:' '3 x 128 x 128')
# print('Validation Split:', validation_split)
print('device:', device)

# Confusion Matrix  4 X 4 , classes row/coloumn
print('Confusion Matrix')
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(validation_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)






## Trash 

# @torch.no_grad()
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([])
#     for batch in loader:
#         images, labels = batch

#         preds = model(images)
#         all_preds = torch.cat(
#             (all_preds, preds)
#             ,dim=0
#         )
#     return all_preds
# with torch.no_grad():
#     prediction_loader = torch.utils.data.DataLoader(train_loader, batch_size=256)
#     train_preds = get_all_preds(model, prediction_loader)


# preds_correct = get_num_correct(train_preds, train_set.targets)

# print('total correct:', preds_correct)
# print('accuracy:', preds_correct / len(train_set) * 100)

# train_set.targets
# train_preds.argmax(dim=1)

# stacked = torch.stack(
#     (
#         train_set.targets
#         ,train_preds.argmax(dim=1)
#     )
#     ,dim=1
# )

# stacked.shape

# stacked

# stacked[0].tolist()

# cmt = torch.zeros(4,4, dtype=torch.int64)

# for p in stacked:
#     tl, pl = p.tolist()
#     cmt[tl, pl] = cmt[tl, pl] + 1
# cmt

# import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix
# from resources.plotcm import plot_confusion_matrix

# cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
# print(type(cm))
# cm

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# from plotcm import plot_confusion_matrix

# plt.figure(figsize=(4, 4))
# plot_confusion_matrix(cm, train_set.classes)

# torch.save(arg, path)
# torch.load(path)
# model.load_state_dict(arg)

# torch.save(model.state_dict(), PATH)


# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()



# Adaptive learning rate
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001 ) # momentum=0.1

# step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.1)

# writer.add_graph(model, images)
 




# torch.save(model.state_dict(), PATH)

# torch.save({
#             'epoch': num_epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss            
#             }, PATH)


# # Loading the saved model and ckpts 

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss'] 


# with torch.no_grad():
#     n_correct = 0
#     n_sample = 0
#     n_class_correct = [0 for i in range(10)]
#     n_class_samples = [0 for i in range(10)]

#     for images, labels in enumerate(test_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)

#         # max return(value, index)
#         _, predicted = torch.max(outputs,1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#         for i in range(batch_size):
#             label = label[i]
#             pred = predicted[i]
#             if (label == pred):
#                 n_class_correct[label] += 1
#             n_class_samples[label] += 1    
#     acc = 100 * n_correct / n_sample
#     print(f'Accuracy of Network:, {acc} %')