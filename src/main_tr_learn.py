# transfer_learning

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms

# import matplotlib.plt as plt 
import time
import os
import copy
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.models as models 

# device = torch.device('cuda' if torch.cuda.is_avalable() else 'cpu ')

# mean = np.arrary([0.485, 0.456, 0.406])
# std = np.arrary([0.229, 0.224, 0.225])

# data_transform = {'train': transforms.Compose([transforms.RandomResizeCrop(224),
# 						transforms.RandomHorigontalFlip(),
# 						transforms.ToTensor()])}
def set_parameter_requires_grad(model, extracting):

	if extracting:
		for parms in model.parameters():
			parms.requires_grad = False





def train_model(model, data_loader, criterion, optimizer, epochs):
	for epoch in range(epochs):
		print('Epoch %d / %d' % (epoch, epochs-1))
		print('-'*15)

		for phase in ['train', 'validation']:
			if phase  == 'train':
				model.train()

			else:
				model.eval()

			running_loss = 0.0
			correct = 0

			for inputs, labels in data_loader[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()


				with T.set_grad_enabled(phase=='train'):
					outputs = model(inputs)

					loss = criterion(outputs, labels)

					_, preds = T.max(outputs,1)

					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				correct += T.sum(preds == labels.data)
			epoch_loss = running_loss /len(data_loader[phase].dataset)
			epoch_acc = correct.double() / len(data_loader[phase].dataset)

			print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
if __name__ == '__main__':
	root_dir = 'seeds_dataset/data1/'
	epochs = 100
	image_transforms = {
	'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
	'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
	}
	data_generator = {k: datasets.ImageFolder(os.path.join(root_dir, k),
                                          image_transforms[k])
                  for k in ['train', 'validation']}
	data_loader = {k: T.utils.data.DataLoader(data_generator[k], batch_size=128,
		shuffle=True, num_workers=4) for k in ['train', 'validation']}

	device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
	model = models.resnet18(pretrained=True)

	set_parameter_requires_grad(model, True)

	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 4)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
	# optimizer = optim.Adam(model.parameters(), lr=0.001)

	parms_to_update = []

	for name, parms in model.named_parameters():
		parms_to_update.append(parms)
		# print('\t', name)

	train_model(model, data_loader, criterion, optimizer, exp_lr_scheduler, epochs)



