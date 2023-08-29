import numpy as np
import os
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops

import matplotlib.pyplot as plt
import cv2


class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

def load_and_prepare_model(model_path):
    # we are going to register a PyTorch hook on the important parts of the YOLO model,
    # then reverse engineer the outputs to get boxes and logits
    # first, we have to register the hooks to the model *before running inference*
    # then, when inference is run, the hooks will save the inputs/outputs of their respective modules
    model = YOLO(model_path)
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    detect_hook = SaveIO()
    for i, module in enumerate(model.model.modules()):
        if type(module) is Detect:
            module.register_forward_hook(detect_hook)
            detect = module

            cv2_hooks = [SaveIO() for _ in range(module.nl)]
            cv3_hooks = [SaveIO() for _ in range(module.nl)]
            for i in range(module.nl):
                module.cv2[i].register_forward_hook(cv2_hooks[i])
                module.cv3[i].register_forward_hook(cv3_hooks[i])
            break
    input_hook = SaveIO()
    model.model.register_forward_hook(input_hook)

    # save and return these for later
    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks]

    return model, hooks


def plot_image(img_path, results):
    img = cv2.imread(img_path)
    for box in results:
        x0, y0, x1, y1 = [int(b) for b in box['bbox']]
        # print("new bounding" +  str(box['bbox']))
        img = cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),3)

    plt.imshow(img[:,:,::-1])
    plt.savefig(f'{os.path.basename(img_path)}_test.jpg')


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Bounding box coordinates [x1, y1, w1, h1].
        box2 (list): Bounding box coordinates [x2, y2, w2, h2].

    Returns:
        float: Intersection over Union (IoU) value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_area = max(0, intersect_x2 - intersect_x1 + 1) * max(0, intersect_y2 - intersect_y1 + 1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersect_area / float(box1_area + box2_area - intersect_area)
    return iou


# Apply Non-Maximum Suppression
def nms(boxes, iou_threshold=0.7):
    """
    Applies Non-Maximum Suppression (NMS) to a list of bounding box dictionaries.

    Args:
        boxes (list): List of dictionaries, each containing 'bbox', 'logits', and 'activations'.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Sort boxes by confidence score in descending order
    sorted_boxes = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)
    selected_boxes = []

    # Keep the box with highest confidence and remove overlapping boxes
    while sorted_boxes:
        delete_idxs = []
        for i, box0 in enumerate(sorted_boxes):
            for j, box1 in enumerate(sorted_boxes):
                if i < j and calculate_iou(box0['bbox'], box1['bbox']) > iou_threshold:
                    delete_idxs.append(j)

        # now delete by popping them in reverse order
        filtered_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in delete_idxs]

    return filtered_boxes


def run_predict(img_path, model, hooks, threshold=0.5, iou=0.7, save_image = False):
    """Run prediction with a YOLO model and get logits/class scores.
    Args:
        img_path: path to an image file
        model: a YOLO object (see load_and_prepare_model() function above)
        hooks: hooks added by the load_and_prepare_model() function above
    Returns
        boxes: a list of dictionaries. each dictionary contains:
            bbox: list, [x0,y1,x1,y1], in original image coordinate space
            logits: list, raw logits vector, one entry per class
            activations: list, pred scores after calling logits.sigmoid()
    """
    # unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # run inference; we don't actually need to store the results because
    # the hooks store everything we need
    model(img_path)

    # now reverse engineer the outputs to find the logits
    # see Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, cls = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    # assumes batch size = 1 (i.e. you are just running with one image)
    # if you want to run with many images, throw this in a loop
    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = cls[batch_idx]

    # figure out the original img shape and model img shape so we can transform the boxes
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # compute predictions
    boxes = []
    for i in range(xywh_sigmoid.shape[-1]): # for each predicted box...
        x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:,i]
        x0, y0, x1, y1 = ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
        logits = all_logits[:,i]
        
        if max(class_probs_after_sigmoid) > threshold:
            boxes.append({
                'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],
                'logits': logits.cpu().tolist(),
                'activations': [p.item() for p in class_probs_after_sigmoid]
            })

    nms_results = nms(boxes, iou)

    if save_image:
        plot_image(img_path, nms_results)

    return nms_results


### Start example script here ###
### (This shows how to use the methods in this file) ###
def main():
    # change these, of course :)
    SAVE_TEST_IMG = False
    model_path = 'yolov8n.pt'
    img_path = 'bus.jpg'
    threshold = 0.5

    # load the model
    model, hooks = load_and_prepare_model(model_path)

    # run inference
    results = run_predict(img_path, model, hooks)

    print("Processed", len(results), "boxes")
    print("The first one is", results[0])

    if SAVE_TEST_IMG:
        plot_image(img_path, results)

if __name__ == '__main__':
    main()