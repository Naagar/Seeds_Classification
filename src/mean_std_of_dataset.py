import torch
from torchvision import datasets, transforms

from load_data import seeds_dataset                          # load the data Set


dataset = datasets.ImageFolder('train',
                 transform=transforms.ToTensor())

batch_size = 64  

train_txt_path='seeds_dataset/data/train_data_file.csv' 
train_img_dir='seeds_dataset/data/train'
test_text_path='seeds_dataset/data/test_data_file.csv'
test_img_dir='seeds_dataset/data/validation'

# Load dataset. train and test
train_dataset = seeds_dataset(train_txt_path,train_img_dir)
test_dataset = seeds_dataset(test_text_path,test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
validation_loader = DataLoader(test_dataset, batch_size=batch_size)


mean = 0.
std = 0.
for images, _ in train_loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(train_loader.dataset)
std /= len(train_loader.dataset)

print(f'Mean: {mean}')

print(f'Std: {std}')
