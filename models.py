## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        
        # 32 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (64, 106, 106)
        # after one pool layer, this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # 64 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (64, 49, 49)
        # after one pool layer, this becomes (64, 24, 24)
        self.conv3 = nn.Conv2d(64, 64, 5)
        
        # 64 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        # the output Tensor for one image, will have the dimensions: (32, 22, 22)
        # after one pool layer, this becomes (64, 11, 11)
        self.conv4 = nn.Conv2d(64, 32, 3)
        
        # 64 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (11-3)/1 +1 = 9
        # the output Tensor for one image, will have the dimensions: (32, 9, 9)
        # after one pool layer, this becomes (32, 4, 4)
        self.conv5 = nn.Conv2d(32, 32, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 outputs * the 4*4 filtered/pooled map size
        self.fc1 = nn.Linear(32*4*4, 256)
        
        # dropout with p=0.5
        self.drop = nn.Dropout(p=0.5)
        
        # finally, create 136 output channels
        self.fc2 = nn.Linear(256, 136)
        
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))
        
        x = self.pool(F.relu(self.conv5(x)))
        
        # prep for linear layer
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
class Net_2(nn.Module):

    def __init__(self):
        super(Net_2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        
        # 32 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (64, 106, 106)
        # after one pool layer, this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # 64 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output Tensor for one image, will have the dimensions: (64, 51, 51)
        # after one pool layer, this becomes (64, 25, 25)
        self.conv3 = nn.Conv2d(64, 64, 3)
        
        # 64 input image channel (grayscale), 128 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (25-3)/1 +1 = 23
        # the output Tensor for one image, will have the dimensions: (128, 23, 23)
        # after one pool layer, this becomes (128, 11, 11)
        self.conv4 = nn.Conv2d(64, 128, 3)
        
        # 128 input image channel (grayscale), 128 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (11-3)/1 +1 = 9
        # the output Tensor for one image, will have the dimensions: (128, 9, 9)
        # after one pool layer, this becomes (128, 4, 4)
        self.conv5 = nn.Conv2d(128, 128, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 128 outputs * the 4*4 filtered/pooled map size
        self.fc1 = nn.Linear(128*4*4, 512)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # Create 256 output channels
        self.fc2 = nn.Linear(512, 256)
        
        # dropout with p=0.4
        self.fc2_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels
        self.fc3 = nn.Linear(256, 136)
        
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))
        
        x = self.pool(F.relu(self.conv5(x)))
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (16, 220, 220)
        # after one pool layer, this becomes (16, 110, 110)
        self.conv1 = nn.Conv2d(1, 16, 5) 
        
        # 16 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (32, 106, 106)
        # after one pool layer, this becomes (32, 53, 53)
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        # 32 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (32, 49, 49)
        # after one pool layer, this becomes (32, 24, 24)
        self.conv3 = nn.Conv2d(32, 32, 5)
        
        # 32 input image channel (grayscale), 16 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        # the output Tensor for one image, will have the dimensions: (16, 22, 22)
        # after one pool layer, this becomes (16, 11, 11)
        self.conv4 = nn.Conv2d(32, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        # 16 outputs * the 11*11 filtered/pooled map size
        self.fc1 = nn.Linear(16*11*11, 512)
        
        # dropout with p=0.5
        self.drop = nn.Dropout(p=0.5)
        
        # fc2 with input=512 and ouput=256
        self.fc2 = nn.Linear(512, 256)
        
        # finally, create 136 output channels
        self.fc3 = nn.Linear(256, 136)
        
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x    