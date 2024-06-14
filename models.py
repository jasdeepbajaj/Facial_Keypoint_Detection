import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define the convolutional layers and their corresponding parameters.
        
        # Layer 1: 
        # - Input: 1 channel (grayscale image)
        # - Output: 32 feature maps
        # - Kernel size: 5x5
        # - Followed by BatchNorm, MaxPooling and Dropout
        self.conv1 = nn.Conv2d(1, 32, 5)
        # self.bn1 = nn.BatchNorm2d(32)  # Batch normalization added here
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        
        # Layer 2:
        # - Input: 32 feature maps
        # - Output: 64 feature maps
        # - Kernel size: 3x3
        # - Followed by BatchNorm, MaxPooling and Dropout
        self.conv2 = nn.Conv2d(32, 64, 3)
        # self.bn2 = nn.BatchNorm2d(64)  # Batch normalization added here
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.2)
        
        # Layer 3:
        # - Input: 64 feature maps
        # - Output: 128 feature maps
        # - Kernel size: 3x3
        # - Followed by BatchNorm, MaxPooling and Dropout
        self.conv3 = nn.Conv2d(64, 128, 3)
        # self.bn3 = nn.BatchNorm2d(128)  # Batch normalization added here
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.3)
        
        # Layer 4:
        # - Input: 128 feature maps
        # - Output: 256 feature maps
        # - Kernel size: 3x3
        # - Followed by BatchNorm, MaxPooling and Dropout
        self.conv4 = nn.Conv2d(128, 256, 3)
        # self.bn4 = nn.BatchNorm2d(256)  # Batch normalization added here
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.4)
        
        # Layer 5:
        # - Input: 256 feature maps
        # - Output: 512 feature maps
        # - Kernel size: 3x3
        # - Followed by BatchNorm, MaxPooling and Dropout
        self.conv5 = nn.Conv2d(256, 512, 3)
        # self.bn5 = nn.BatchNorm2d(512)  # Batch normalization added here
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0.5)
        
        # Fully Connected Layers:
        
        # Layer 6:
        # - Input: Flattened 512 feature maps (512*5*5)
        # - Output: 2560 neurons
        # - Followed by BatchNorm and Dropout
        self.fc6 = nn.Linear(512 * 5 * 5, 2560)
        # self.bn6 = nn.BatchNorm1d(2560)  # Batch normalization added here
        self.drop6 = nn.Dropout(p=0.4)
        
        # Layer 7:
        # - Input: 2560 neurons
        # - Output: 1280 neurons
        # - Followed by BatchNorm and Dropout
        self.fc7 = nn.Linear(2560, 1280)
        # self.bn7 = nn.BatchNorm1d(1280)  # Batch normalization added here
        self.drop7 = nn.Dropout(p=0.4)
        
        # Layer 8:
        # - Input: 1280 neurons
        # - Output: 136 neurons (representing 68 keypoints with (x, y) pairs)
        self.fc8 = nn.Linear(1280, 136)
        

    def forward(self, x):
        # Define the forward pass
        
        # Apply first conv layer, followed by BatchNorm, ReLU, max pooling, and dropout
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        # Apply second conv layer, followed by BatchNorm, ReLU, max pooling, and dropout
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        # Apply third conv layer, followed by BatchNorm, ReLU, max pooling, and dropout
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        # Apply fourth conv layer, followed by BatchNorm, ReLU, max pooling, and dropout
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        # Apply fifth conv layer, followed by BatchNorm, ReLU, max pooling, and dropout
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer, followed by BatchNorm, ReLU and dropout
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        
        # Apply second fully connected layer, followed by BatchNorm, ReLU and dropout
        x = F.relu(self.fc7(x))
        x = self.drop7(x)
        
        # Output layer without activation function (regression output)
        x = self.fc8(x)
        
        return x
