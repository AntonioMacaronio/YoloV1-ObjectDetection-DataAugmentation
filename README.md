# Object Detection with YoloV1 
This repository contains an implementation of YOLO (You Only Look Once) Object Detection which uses a deep convolutional neural network. Object detection is the problem involving when given an image, we would like to identify all the objects in the image and draw a bounding box around them.

The dataset used for training is Pascal VOC, located here: http://host.robots.ox.ac.uk/pascal/VOC/

# General Algorithm of YoloV1
1. YoloV1 uses an architecture coined 'darknet' by authors, which is a deep convolutional neural network with maxpooling and many filters to learn spatial localities of an image.
2. Yolo splits up an image into an SxS grid of cells; in this repository, I've chosen S=7.
3. Each cell will predict B=2 bounding boxes, which a length 30 tuple with [0:19] = 20 classes of objects, [20:24] = 1st bounding box, [25:29] = 2nd bounding box (these indices are inclusive on both ends)
4. In the end, the model will output a `predictions_matrix`, which is of shape (batchSize, S * S * 30)
5. We use non-max suppression to filter `predictions_matrix` to get our final bounding box
6. Then, we pass all of this information into a loss function based off of the sum of squared differences (SSD) error for backpropogation. The loss function penalizes the bounding boxes "responsible" for a cell (A bounding box is "responsible" for a cell if it has the highest IoU out of all the boxes in that cell.


# Key Ideas
### Non-Max Suppression
Problem to Solve: We have too many bounding boxes, and we need to suppress


### Intersection Over Union (IoU)
We need a metric to determine how much too bounding boxes overlap

### Mean Average Precision (mAP)
To quantify the performance of an object detection model, 

# Results



# Yolo Struggles and Improvements:
1. YoloV1 heavily struggles on small objects that are densely packed, such as a flock of birds. Because it really only outputs 2 bounding boxes per cell, it is limited to detecting S*S objects at most.
2. Data Augmentation - Yolo struggles to generalize to 
