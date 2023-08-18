from ultralytics import YOLO
from constants import *

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=DATASET_YAML, 
                      epochs = 3
                      )

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')