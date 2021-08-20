"""
The labels folder has text files containing the following:
class, center(x,y) which are from [0,1], length, width

Because each image has a different pixel size, we will have to process these
Each image in the images folder has corresponding .txt file with same name
"""

import torch
import os
import pandas as import pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S = 7, B = 2, C = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # class_label is an integer, but other values are likely to be floats. 
                # therefore, we take the float of x if it does not equal its integer part
                # otherwise, we take the integer part (because it is an integer)
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label) #making sure its integer, avoids rounding errors
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S*x - j, self.S*y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
        
