import torch.nn as nn
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F


num_classes = 4   #   output  classes
batch_size = 32
#  img = size[3, 256, 256]

# input_size  = [batch_size,no_ch, 265, 256]
# Create a neural net class
class seeds_model(nn.Module):
    # Constructor
    def __init__(self, num_classes=4):
        super(seeds_model, self).__init__()
        
        # Our images are RGB, so input channels = 3. wW'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 256x256 image tensors will be pooled twice with a kernel size of 2. 256/2/2 is 64.
        # So our feature tensors are now 64 x 64, and we've generated 24 of them
        # We need to flatten these to map them to  the probability for each class
        self.fc = nn.Linear(in_features=63 * 63 * 24, out_features=num_classes)   # fulley connected layer

    def forward(self, x):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
        # Use a relu activation function after layer 1 (convolution 2 and drop)
        
        # Use a relu activation function after layer 3 (convolution 3)
        x = F.relu(self.pool(self.conv2(x)))
        
        # Drop some features after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        
        # Flatten
        x = x.view(-1, 63 * 63 * 24)
        x = self.fc(x)
        # Return class probabilities via a softmax function 
        return F.log_softmax(x, dim=1)
    


# model = seeds_model()

