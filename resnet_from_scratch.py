import torch
import torch.nn 
 
class block(nn.Module):
	def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):

		super(block,self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
		self.relu = nn.ReLU()
		self.identity_downsample = identity_downsample

	def forward(self, x):
		indetity = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		# x = self.conv1(x)
		# x = self.bn1(x)

		if self.identity_downsample is not None:
			indetity = self.identity_downsample(indetity)

		x += indetity
		x = self.relu(x)
		return x

class ResNet(nn.Module):  # [3, 4, 6, 3]
	def __init__(self, block, layers, image_channels, num_classes):
		super(ResNet, self).__init__()
		self.image_channels = 64
		self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def make_layer(slef, block, num_residual_blocks, out_channels, stride):
		identity_downsample = None
		layers = []

		if stride != 1 or slef.image_channels != out_channels *4 :
			identity_downsample =nn.Sequential(nn.Conv2d(self.image_channels, out_channels*4, kernel_size=1,
														stride=stride),
												nn.BatchNorm2d(out_channels*4))
			


		# resnet layers

		self.layer1 =




