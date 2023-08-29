import numpy as np
import os
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image

import json


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


def is_text_file(file_path):
    # Check if the file extension indicates a text file
    text_extensions = ['.txt'] #, '.csv', '.json', '.xml']  # Add more extensions if needed
    return any(file_path.lower().endswith(ext) for ext in text_extensions)

def is_image_file(file_path):
    # Check if the file extension indicates an image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def plot_image(img_path, results, category_mapping = None):
    """
    Display the image with bounding boxes and their corresponding class scores.

    Args:
        img_path (str): Path to the image file.
        results (list): List of dictionaries containing bounding box information.

    Returns:
        None
    """

    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box in results:
        x0, y0, x1, y1 = map(int, box['bbox'])
        box_color = "r"  # red
        tag_color = "k"  # black
        max_score = max(box['activations'])
        max_category_id = box['activations'].index(max_score)
        category_name = max_category_id

        if category_mapping:
            max_category_name = category_mapping.get(max_category_id, "Unknown")
            category_name = max_category_name

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=box_color,
            label=f"{max_category_id}: {category_name} ({max_score:.2f})",
            facecolor='none'
        )
        ax.add_patch(rect)

        plt.text(
            x0,
            y0 - 50,
            f"{max_category_id} ({max_score:.2f})",
            fontsize="5",
            color=tag_color,
            backgroundcolor=box_color,
        )

    ax.legend(fontsize="5")
    plt.axis("off")
    plt.savefig(f'{os.path.basename(img_path)}_test.jpg', bbox_inches="tight", dpi=300)


def write_json(img_path, results):
    # Create a list to store the predictions data
    predictions = []

    for result in results:
        image_id = os.path.basename(img_path).split('.')[0]
        max_category_id = result['activations'].index(max(result['activations']))
        category_id = max_category_id
        bbox = result['bbox']
        score = max(result['activations'])
        activations = result['activations']

        prediction = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score,
            'activations': activations
        }

        predictions.append(prediction)

    # Write the predictions list to a JSON file
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)


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
    delete_idxs = []
    for i, box0 in enumerate(sorted_boxes):
        for j, box1 in enumerate(sorted_boxes):
            if i < j and calculate_iou(box0['bbox'], box1['bbox']) > iou_threshold:
                delete_idxs.append(j)

    # Reverse the order of delete_idxs
    delete_idxs.reverse()

    # now delete by popping them in reverse order
    filtered_boxes = [box for idx, box in enumerate(sorted_boxes) if idx not in delete_idxs]

    return filtered_boxes


def results_predict(img_path, model, hooks, threshold=0.5, iou=0.7, save_image = False, category_mapping = None):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.
        save_image (bool, optional): Whether to save the image with boxes plotted. Default is False.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
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
        
        # Filter by score threshold (of max score class)
        if max(class_probs_after_sigmoid) > threshold:
            boxes.append({
                'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],
                'logits': logits.cpu().tolist(),
                'activations': [p.item() for p in class_probs_after_sigmoid]
            })

    nms_results = nms(boxes, iou)

    if save_image:
        plot_image(img_path, nms_results, category_mapping)

    # if save_json:
    #     write_json(img_path, nms_results)

    return nms_results


def run_predict(input_path, model, hooks, threshold=0.5, iou_threshold=0.7, save_image = False, save_json = False, category_mapping = None):
    """
    Run prediction with a YOLO model.

    Args:
        input_path (str): Path to an image file or txt file containing paths to image files.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.
        save_image (bool, optional): Whether to save the image with boxes plotted. Default is False.
        save_json (bool, optional): Whether to save the results in a json file. Default is False.

    Returns:
        list: List of selected bounding box dictionaries for all the images given as input.
    """
    use_txt_input = False

    if is_text_file(input_path):
        use_txt_input = True

    if use_txt_input:
        with open(input_path, 'r') as f:
            img_paths = f.read().splitlines()
    else:
        img_paths = [input_path]

    all_results = []

    for img_path in img_paths:
        results = results_predict(img_path, model, hooks, threshold, iou=iou_threshold, save_image=save_image, category_mapping=category_mapping)

        all_results.extend(results)

    if save_json:
        write_json(img_path, all_results)

    return all_results


### Start example script here ###
### (This shows how to use the methods in this file) ###
def main():
    # change these, of course :)
    SAVE_TEST_IMG = False
    model_path = 'yolov8n.pt'
    img_path = 'bus.jpg'
    threshold = 0.5
    iou_threshold = 0.7

    # category mapping to show category name in addition of category id
    category_mapping = {
        0: "bird",
        1: "cow",
        2: "domestic dog",
        3: "egyptian mongoose",
        4: "european badger",
        5: "european rabbit",
        6: "fallow deer",
        7: "genet",
        8: "horse",
        9: "human",
        10: "iberian hare",
        11: "iberian lynx",
        12: "red deer",
        13: "red fox",
        14: "wild boar",
    }

    # or not category_mapping (=None)

    # load the model
    model, hooks = load_and_prepare_model(model_path)

    # run inference
    results = run_predict(img_path, model, hooks, threshold, iou=iou_threshold, save_image=SAVE_TEST_IMG, save_json=True, category_mapping=category_mapping)

    # Print Boxes information
    print("Processed", len(results), "boxes")

    for result in results:
        print("\n")
        print("Bounding Box :" + str(result['bbox']))
        print("Logits :" + str(result['logits']))
        print("Activations :" + str(result['activations']))
        print("\n")

if __name__ == '__main__':
    main()