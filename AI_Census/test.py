import comet_ml

from ultralytics import YOLO
from constants import *
from constants_unshared import MY_API_KEY_COMET
import os

os.environ["COMET_API_KEY"] = MY_API_KEY_COMET

PATH = "cv4ecology/"

# initialize experiment in comet
#comet_ml.init("ai-census") # it get the name of the project name on training

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS_BEST)

metrics = model.val(split="test",           # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
                    save_json = True,       # (bool) save results to JSON file
                    device = 0,
                    conf = 0.5,              # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                    #save_hybrid = True,   # (bool) save hybrid version of labels (labels + additional predictions)

                    #iou: 0.7                # (float) intersection over union (IoU) threshold for NMS
                    #max_det: 300            # (int) maximum number of detections per image
                    #half: False             # (bool) use half precision (FP16)
                    #dnn: False              # (bool) use OpenCV DNN for ONNX inference
                    #plots: True             # (bool) save plots during train/val
                    )