"""
Loss function is the sum of 4 quantities: 
    1. distance to center
    2. sum of width and height differences from actual 
        - we will take square root to equalize size of bounding boxes, small count the same as large
    3. probability that there is a box
    4. probability that there is NOT a box (no object in the square)

    For all of these quantities, we take the largest IOU
    To maintain convexity so that our numerical descent work, I used L2 Loss
"""
import torch
import torch.nn as nn
from utils import intersection_over_union
