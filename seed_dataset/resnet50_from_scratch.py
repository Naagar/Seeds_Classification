#  resnet50_from_scratch
import torch
import torch.nn as nn

class block(nn.Module):
	def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
		super(block,self).__init__()
		self.expension = 4
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1,padding=0)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=stride,padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expension, kernel_size=1,stride=1,padding=0)
		self.bn3 = nn.BatchNorm2d(out_channels*self.expension)
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
		
		if self.identity_downsample is not None:
			indetity = self.identity_downsample(indetity)
		x +=  indetity
		x = self.relu(x)
		return x

class resnet(nn.Module):   #  [3, 4, 6, 3]
	def __init__(self, block, layers, image_channels, num_classes):
		super(resnet,self).__init__()
		self.in_channels = 64 
		self.conv1 =nn.Conv2d(image_channels, 64, kernel_size =7, stride=2, padding=3)
		self.bn1  = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



		# resnet layers

		self.layers1 = self.make_layers(block, layers[0], out_channels=64, stride=1)
		# self.layers2 =  ...
		self.layers2 = self.make_layers(block, layers[1], out_channels=128, stride=2)
		self.layers3 = self.make_layers(block, layers[2], out_channels=256, stride=2)
		self.layers4 = self.make_layers(block, layers[3], out_channels=512, stride=2)
		

		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(512*4, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layers1(x)
		x = self.layers2(x)
		x = self.layers3(x)
		x = self.layers4(x)

		x = self.avg_pool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)
		return x



	def make_layers(self, block, num_residual_blocks, out_channels, stride):
		identity_downsample =None

		layers = []

		if stride != 1 or self.in_channels != out_channels * 4:
			identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
															stride=stride),
															nn.BatchNorm2d(out_channels*4))
		layers.append(block(self.in_channels, out_channels, identity_downsample, stride))   ## fisrt block going to chnage nom_channels
		self.image_channels = out_channels*4

		for i in range(num_residual_blocks - 1):
			layers.append(block(self.in_channels, out_channels))  # 256  -> 64, 64 * 4 (256) again


		return nn.Sequential(*layers)


class resnet50(image_channels=3, num_classes=4):
	return resnet(block, [3, 4, 6, 3], image_channels, num_classes)

class resnet101(image_channels=3, num_classes=4):
	return resnet(block, [3, 4, 23, 3], image_channels, num_classes)

class resnet152(image_channels=3, num_classes=4):
	return resnet(block, [3, 8, 36, 3], image_channels, num_classes)

def test():
	net = resnet50()
	x = torch.randn(2, 3, 224, 224)
	y = net(x).to(cuda)
	print(y.shape)

test()