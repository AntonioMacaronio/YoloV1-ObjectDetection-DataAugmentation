import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class YoloV1_Pretrained(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        """
        Keyword Arguments (**kwargs):
            1. S = how many split boxes you chop up the original image into (7x7 in this case)
            2. B = how many bounding boxes we are computing per split box (we will have 2 boxes)
            3. C = number of classes (in our dataset, there are 20)
        """
        super().__init__()
        self.S =  kwargs["S"]
        self.B =  kwargs["B"]
        self.C =  kwargs["C"]
        self.output_dim = self.C + self.B * 5
        
        pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        pretrained.requires_grad_(False)
        pretrained.avgpool = nn.Identity()
        pretrained.fc = nn.Identity()
        
        self.layer1 = pretrained
        
        num_internal_channel = 1024
        self.layer2 = nn.Sequential(
            nn.Conv2d(2048, num_internal_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(num_internal_channel, num_internal_channel, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(num_internal_channel, num_internal_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(num_internal_channel, num_internal_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * num_internal_channel, 4096),
            nn.Dropout(0.1), # may need to comment this out
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096,  self.S * self.S * self.output_dim)
        )

    def forward(self, x):
        """
        Input:
            - 1. x = torch.size([batchSize, 3, 448, 448]) tensor which represents an image with RGB channel
        Output:
            - 1. pred = torch.size([batchSize, 30 * S * S]) tensor, 20 classes + 5 dims for bbox 1 + 5 dim for bbox 2
        Notes:
            - for each bbox, it has 5 dim: (probability, x_center (numpyCol), y_center (numpyRow), length_in_x (numCols), length_in_y (numRows))
        """
        x = self.layer1(x)
        x = x.reshape(-1, 2048, 14, 14)
        return self.layer2(x)


def test(S = 7, B = 2, C = 20): 
    """
    Test Parameters:
        1. S = how many boxes you chop up the original image into
        2. B = how many bounding boxes we are computing
        3. C = number of classes
    """
    model = YoloV1_Pretrained(S = S, B = B, C = C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

# test()

