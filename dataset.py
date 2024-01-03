"""
The labels folder has text files containing the following:
class, center(x,y) which are from [0,1], length, width

Because each image has a different pixel size, we will have to process these
Each image in the images folder has corresponding .txt file with same name

Pytorch is beautiful because to create a dataset, we only need to specify how to get 1 example from the dataset. 
The Pytorch framework will handle how to batch this and dataloading for us.
"""

import torch
from torchvision.transforms.functional import pil_to_tensor
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file) # Note: this a pandas dataframe with each row representing a datapoint. each row has 2 cols: jpg_fname, labeltxt_fname
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform # there will be a transform passed in, which resizes an image
        self.S = S
        self.B = B
        self.C = C
    
    def __len__(self):
        """Returns the number of datapoints"""
        return len(self.annotations)

    def __getitem__(self, index):
        """Gets a image and a label from the dataset
        Outputs:
            1. image = tensor of shape (3, 448, 448)
            2. label_matrix = tensor of shape (S, S, 30) where first 2 dimensions specify the cell, and last dim specifies the cell bbox
        Notes:
            'boxes' is converted into 'label_matrix', where boxes looks like [[class_label, x, y, width, height]]
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = [] # there could be multiple bboxes
        with open(label_path) as f:
            for label in f.readlines():
                # class_label is an integer, but other values are likely to be floats. therefore, we take the float of x if it does not equal its integer part otherwise, we take the integer part (because it is an integer)
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
                # NOTE: example: 000001.txt file has each line with [class_enum, center_x (numpyRow), center_y (numpyCol), width, height] 
                # last 4 values are relative to the image, so they're between 0 and 1
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes) # boxes has shape [numBoxes, 5]

        if self.transform: 
            # image, boxes = self.transform(image, boxes)
            image = self.transform(image) # they now have shape (3, 448, 448)
        
        # TODO: fix this label_matrix so that it is (S, S, 25)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B)) # shape (S, S, 30) tensor that contains the labels for each cell, but last 5 are not used
        # for every bbox from the label.txt file, we find what cell it's in our label matrix, and set that equal to 1
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label) # making sure its integer, avoids rounding errors
            i, j = int(self.S * y), int(self.S * x) # these are the coordinates of the cell (rmbr, we split a photo into SxS cells)
            x_cell, y_cell = self.S*x - j, self.S*y - i # now we have the x and y coordinates with respect to that cell (between 0 and 1)
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) # x_cell, y_cell are in [0, 1]
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix  
        
