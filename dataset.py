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