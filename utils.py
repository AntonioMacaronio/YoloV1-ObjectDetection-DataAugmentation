"""
1. Intersection over Union
2. No-Max Supression
3. Mean Avg Precision
4. Converting from relative cell to entire image

"""

import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """Calculates intersection over union
    Input:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Output:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="midpoint"):
    """Does Non Max Suppression given bboxes for a specific class (purpose is to reduce # of bboxes in a cell)
    Input:
        1. bboxes (list): list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
        2. iou_threshold (float): threshold where predicted bboxes is correct
        3. threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        4. box_format (str): "midpoint" or "corners" used to specify bboxes
    Output:
        1. bboxes_after_nms (list): bboxes after performing NMS given a specific IoU threshold

    Notes:
        - General Algorithm:
            1. Discard all bboxes < probability threshold
            2. For the largest probability bbox, and remove those that have IOU > iou_threshold
        - This method is called for every cell (doesn't mix bboxes from separate cells)
    """

    assert type(bboxes) == list
    # print(len(bboxes))
    bboxes = [box for box in bboxes if box[1] > prob_threshold] # removes all bboxes with low probability of an object
    # print(len(bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # sort the boxes by highest probability score at the beginning
    bboxes_after_nms = []

    while len(bboxes) > 0:
        chosen_box = bboxes.pop(0)
        
        # keeps all the boxes that are not of the same class or have low IOU
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # if it's not of the same class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold # if it's not greater than the threshold, keep it.
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """Calculates mean average precision 
    Input:
        1. pred_boxes (list): list of lists containing all bboxes with each bboxes specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        2. true_boxes (list): Similar as pred_boxes except all the correct ones 
        3. iou_threshold (float): threshold where predicted bboxes is correct
        4. box_format (str): "midpoint" or "corners" used to specify bboxes
        5. num_classes (int): number of classes
    Output:
        1. float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all average precisions for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, classEnum_to_color=None, classEnum_to_className=None):
    """Plots predicted bounding boxes on the image
    Input:
        1. imaage = tensor with shape torch.Size([numRows, numCols, 3])
        2. boxes (list of lists) = [[train_idx, class_prediction, prob_score, x1, y1, x2, y2],...], each list within the big list represents a bbox
    """
    if not isinstance(image, np.ndarray):
        im = np.array(image)
    else:
        im = image
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # A simple heuristic: lighter/brighter colors look better on dark backgrounds
    dark_colors = ['red', 'blue', 'green', 'purple', 'brown', 'maroon', 'slategray', 'navy', 'indigo', 'olive', 'teal']
    light_colors = ['orange', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'gold', 'orchid', 'turquoise']
    colorToBackground = {color: 'white' if color in dark_colors else 'black' for color in classEnum_to_color.values()}

    # Create a Rectangle patch for each box
    for box in boxes:
        if classEnum_to_color != None:
            class_pred = classEnum_to_className[int(box[1])] # this is a string, like "motorbike" or "person"
            class_color = classEnum_to_color[int(box[1])]
        else:
            class_pred = f"label class {box[1]}"
            class_color = "r"

        prob_score = box[2]
        box = box[3:] # remove train_idx, class_prediction, prob_score
        # box[0] is x midpoint, box[2] is width (numpyCol)
        # box[1] is y midpoint, box[3] is height (numpyRow)

        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2 
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1.5,
            edgecolor=class_color,
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    
        # Add annotation text
        annotation_text = f'{class_pred}: {prob_score:.2f}'
        ax.text(
            upper_left_x * width,
            upper_left_y * height,
            annotation_text,
            color=class_color,
            fontsize=8,
            verticalalignment='bottom',
            bbox=dict(facecolor=colorToBackground[class_color], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
        )

    plt.show()


def plot_bbox_and_label(csv_file, datapoint_index, pred_boxes, target_boxes):
    """Plots the model's predicted bounding box(es) on an image as well as the corresonponding label's bounding box(es) on an image for a side by side comparison, Can be used on both 
    Input:
        1. cvs_file = this is either "train.csv" or "test.csv"
        2. datapoint_index = which datapoint in the training or test set to plot
        3. pred_boxes (list of lists) = [[train_idx, class_prediction, prob_score, x1, y1, x2, y2],...], each list within the big list represents a bbox
        4. target_boxes = ^ but 
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 30)) # 1 row, 2 columns
    fname_dataframe = pd.read_csv("data/" + csv_file) # dataframe with our filenames
    image_fname = "data/images/" + fname_dataframe.iloc[datapoint_index, 0]

    # plot the images
    image_numpy = cv2.imread(image_fname)[...,::-1]
    height, width, _ = image_numpy.shape
    ax[0].imshow(image_numpy)
    ax[1].imshow(image_numpy)


    # setup variables to plot bboxes
    classEnum_to_className = {
        0: 'airplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'TVmonitor',
    }

    classEnum_to_color = {
        0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 
        5: 'cyan', 6: 'magenta', 7: 'yellow', 8: 'brown', 9: 'lime',
        10: 'pink', 11: 'teal', 12: 'olive', 13: 'navy', 14: 'indigo',
        15: 'maroon', 16: 'gold', 17: 'orchid', 18: 'turquoise', 19: 'slategray'
    }
    dark_colors = ['red', 'blue', 'green', 'purple', 'brown', 'maroon', 'slategray', 'navy', 'indigo', 'olive', 'teal']
    light_colors = ['orange', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'gold', 'orchid', 'turquoise']
    colorToBackground = {color: 'white' if color in dark_colors else 'black' for color in classEnum_to_color.values()}

    # plot the model's predicted bboxes (on top of left image) and then plot the actual label's bboxes (on top of right image)
    pred_boxes_filtered = [box for box in pred_boxes if box[0] == datapoint_index]
    target_boxes_filtered = [box for box in target_boxes if box[0] == datapoint_index]

    for col_index, boxes in enumerate([pred_boxes_filtered, target_boxes_filtered]):
        for box in boxes:
            class_pred = classEnum_to_className[int(box[1])] # this is a string, like "motorbike" or "person"
            class_color = classEnum_to_color[int(box[1])]

            prob_score = box[2]
            box = box[3:] # remove train_idx, class_prediction, prob_score
            # box[0] is x midpoint, box[2] is width (numpyCol)
            # box[1] is y midpoint, box[3] is height (numpyRow)

            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            upper_left_x = box[0] - box[2] / 2 
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=1.5,
                edgecolor=class_color,
                facecolor="none",
            )
            # Add the patch to the Axes
            ax[col_index].add_patch(rect)
        
            # Add annotation text
            annotation_text = f'{class_pred}: {prob_score:.2f}'
            ax[col_index].text(
                upper_left_x * width,
                upper_left_y * height,
                annotation_text,
                color=class_color,
                fontsize=8,
                verticalalignment='bottom',
                bbox=dict(facecolor=colorToBackground[class_color], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
            )
    plt.show()



def get_bboxes(loader, model, iou_threshold, prob_threshold, pred_format="cells", box_format="midpoint", device="cuda"):
    """Given an unshuffled dataloader of a dataset, this will generate all the bboxes
    Input:
        1. loader
        2. model
    Output:
        1. all_pred_boxes =  [[train_idx, class_prediction, prob_score, x_center, y_center, x_width, y_height], ...], each list within the big list represents a bbox
        2. all_true_boxes =  [[train_idx, class_prediction, prob_score=1, x_center, y_center, x_width, y_height], ...], each list within the big list represents a bbox
    """
    
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    # for x, y in fox_dataloader:
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device) # has shape (batchSize, 3, 448, 448)
        labels = labels.to(device) # has shape (batchSize, S, S, 30)

        with torch.no_grad():
            predictions = model(x) # predictions has shape (batchSize, S * S * 30)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions) # len(bboxes) = 8
        # print(true_bboxes)
        for idx in range(batch_size):
            # print(f"input into non_max_suppresion on datatpoint {idx}", bboxes[idx]) 
            nms_boxes = non_max_suppression(
                bboxes[idx], # len(bboxes[idx]) = 49 (S * S)
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )
            # print("ARE WE GETTING ANYTHING HERE???", len(nms_boxes))
            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    """
    Input:
        1. out = has shape (batchSize, S, S, 30) if it's a label or has shape (batchSize, S * S * 30) if it's a prediction
    Output:
        1. all_bboxes (list) = list with length=batchSize, each entry is a list of all bboxes for that datapoint, where a bbox is a length 5 list with (class, x_center, y_center, x_width, y_height)
    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1) # converted_pred has shape (batchSize, S*S, 30)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]): # for each datapoint in the batch
        bboxes = []

        for bbox_idx in range(S * S):
            # x.item() is a list of 5 numbers: (class, x_center, y_center, x_width, y_height)
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        
        # if ex_idx == 0: # and len(out.shape) == 4:
        #     print("for the first label in this batch, these are the bboxes:")
        #     print(bboxes)
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])