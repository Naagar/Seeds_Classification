import pandas as pd
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

csv_path = 'seeds_dataset_labels_file.csv'
root_dir = 'seeds_dataset/images/'

class DatasetFromCsvLocation(Dataset):
    def __init__(self, csv_path):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files

        Args:
            csv_path (string): path to csv file
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        # print(root_dir + self.data_info.iloc[:, 0])
        self.image_arr = np.asarray(root_dir + self.data_info.iloc[:, 0])

        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # # Check if there is an operation
        # some_operation = self.operation_arr[index]
        # # If there is an operation
        # if some_operation:
        #     # Do some operation on image
        #     # ...
        #     # ...
        #     pass
        # # Transform image to tensor
        # img_as_img = img_as_img/255
        img_as_tensor = self.to_tensor(img_as_img)


        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # lable = label * [new_w / w, new_h / h]

        return {'image': img, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # landmarks = landmarks - [left, top]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# transformed_dataset = DatasetFromCsvLocation(csv_file='seeds_dataset_labels_file.csv',
#                                            root_dir='seeds_dataset/images',
#                                            transform=transforms.Compose([
#                                                Rescale(256),
#                                                RandomCrop(224)
                                               
#                                           ]))

# def get_data_loader():    
#   # Define transformation to be applied to dataset
#       transform = {
#         'train': transforms.Compose([
#             transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
#             transforms.RandomHorizontalFlip(), # Flip the data horizontally
#             #TODO if it is needed, add the random crop
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize([224,224]),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
#         ])
#     }
  
#   # ImageFloder with root directory and defined transformation methods for batch as well as data augmentation
#   ### Note that ImageFolder is one class which is inherited by torch.utils.data.Dataset
#     data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform['train'] if train else 'test') 
#     data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # return data_loader