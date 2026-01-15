
from torchinfo import summary
import torch.nn as nn


class ModelFullyconnected(nn.Module):

    def __init__(self):
        super(ModelFullyconnected, self).__init__() 

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define layers do modelo
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def forward(self, x):
        # flatten the input to a vector of 1x28x28
        x = x.view(x.size(0), -1)
        # print('Input x.shape = ' + str(x.shape))

        # Now we can pass through the fully connected layer
        y = self.fc(x)
        # print('Output y.shape = ' + str(y.shape))

        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelConvNet(nn.Module):

    def __init__(self):

        super(ModelConvNet, self).__init__()  

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define primeira conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # this will output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output 32x14x14

        # Define segunda conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # output 64x14x14

        # Define segunda pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # this will output 64x7x7

        # Define primeira fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # output 128

        # Define segunda fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        print('Forward method called ...')

        print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        print('After pool2 x.shape = ' + str(x.shape))

        x = x.view(-1, 64*7*7)
        print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        print('Output y.shape = ' + str(y.shape))

        return y


class ModelConvNet3(nn.Module):
    """This is a more complex ConvNet model with 3 conv layers."""

    def __init__(self):

        super(ModelConvNet3, self).__init__()  

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        # Define primeira conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # output 32x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output 32x14x14

        # Define segunda conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # output 64x14x14

        # Define segunda pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output 64x7x7

        # Define segunda conv layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)

        # Define segunda pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define primeira fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        # output 128

        # Define segunda fully connected layer
        self.fc2 = nn.Linear(128, 10)
        # output 10

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.conv1(x)
        # print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        # print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        # print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        # print('After pool2 x.shape = ' + str(x.shape))

        x = self.conv3(x)
        # print('After conv3 x.shape = ' + str(x.shape))

        x = self.pool3(x)
        # print('After pool3 x.shape = ' + str(x.shape))

        # latent vector
        x = x.view(-1, 128*2*2)
        # print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        # print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        # print('Output y.shape = ' + str(y.shape))

        return y
    
    import torch.nn as nn

import torch.nn as nn


class ModelBetterCNN(nn.Module):
    def __init__(self, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  
            nn.Dropout(dropout_conv),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),   
            nn.Dropout(dropout_conv),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, 10)  
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
