## seeds classification

import torch 
import torch.nn as nn
import torchvision.transforms as transforms 
import torch.optim as optim
import torchvision 
from torch.utils.data import DataLoader 
from data_loader import seeds_dataset   						# load the data Set
# from ResNet_models import _resnet50      						##  load resnet model 

#set device 
 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# hyper parameters 

in_channels = 3
num_classes = 4

learning_rate = 1e-3
batch_size = 256
num_epochs = 3


# datalaoder  load data

dataset = seeds_dataset(csv_file = 'seeds_dataset_labels_file.csv', root_dir = 'seeds_dataset/images',
						transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [14000, 3802])

train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size = batch_size, shuffle=True)


model  = torchvision.models.googlenet(pretrained=False)

model.to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optim.Adam(model.parameters(), lr=learning_rate)


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

print('Checking accuracy on Test Set')

check_accuracy(test_loader, model )

