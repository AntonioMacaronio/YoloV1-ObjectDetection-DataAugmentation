import torch
import torch.nn as nn

model_architecture = [
    (7, 64, 2, 3),      # Tuple for Conv Layer: (kernel_size, num_filters, stride, padding)
    "M",                # Maxpool layer
    (3, 192, 1, 1), 
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1), 
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    # List: first_layer, second_layer, num_repetitions_of_prev2_layers
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],   # List: first_layer, second_layer, num_repetitions_of_prev2_layers
    (3, 1024, 1, 1),    # 4 Finishing convolutional layers
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
] 

# this block consists of 3 layers: convolution, batchnorm, and relu
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): # kwargs is variable number of arguments (this will be kernel_size, stride, padding)
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # the reason the bias is False is because we have a BatchNorm (includes bias)
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        """
        Keyword Arguments (**kwargs):
            1. S = how many split boxes you chop up the original image into (7x7 in this case)
            2. B = how many bounding boxes we are computing per split box (we will have 2 boxes)
            3. C = number of classes (in our dataset, there are 20)
        """
        super(YoloV1, self).__init__()
        self.architecture = model_architecture
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture) # darknet is just the name of the conv layers in the paper, we'll implement this
        self.fcs = self.create_fcs(**kwargs) # a final fully connected layer

    def forward(self, x): 
        """
        Input:
            - 1. x = torch.size([batchSize, 3, 448, 448]) tensor which represents an image with RGB channel
        Output:
            - 1. pred = torch.size([30, ]) tensor, 20 classes + 5 dims for bbox 1 + 5 dim for bbox 2
        Notes:
            - for each bbox, it has 5 dim: (probability, x (numpyCol), y (numpyRow), length_in_x, length_in_y)

        """
        x2 = self.darknet(x) # x2 has shape torch.Size([batchSize, 1024, 7, 7])
        x3 = torch.flatten(x2, start_dim=1) # x3 has shape torch.Size([batchSize, 50176]), and we flatten right before the fully-connected layers
        return self.fcs(x3)
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels # represents the current in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels, out_channels = x[1], kernel_size = x[0], stride = x[2], padding = x[3])]
                in_channels = x[1]
            
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            
            elif type(x) == list:
                conv1 = x[0] # first convolutional layer tuple: (kernel_size, num_filters, stride, padding)
                conv2 = x[1] # second convolutional layer tuple
                num_repeats = x[2] 

                for x in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [
                        CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
        print(layers)
        return nn.Sequential(*layers) # here, *layers unpacks the list and sends them all into the Sequential as kwargs

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(1024 * S * S, 496), 
            nn.Dropout(0.0), 
            nn.LeakyReLU(0.1), 
            nn.Linear(496, S*S * (C + B*5)) # this will be reshaped into (S, S, 30) later
        )

def test(S = 7, B = 2, C = 20): 
    """
    Test Parameters:
        1. S = how many boxes you chop up the original image into
        2. B = how many bounding boxes we are computing
        3. C = number of classes
    """
    model = YoloV1(split_size = S, num_boxes = B, num_classes = C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

test()