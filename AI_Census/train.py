import comet_ml

from ultralytics import YOLO
from constants import *
from constants_unshared import MY_API_KEY_COMET
import os

os.environ["COMET_API_KEY"] = MY_API_KEY_COMET
os.environ["COMET_AUTO_LOG_GRAPH"] = "true"
os.environ["COMET_AUTO_LOG_PARAMETERS"] = "true"
os.environ["COMET_AUTO_LOG_METRICS"] = "true"
os.environ["COMET_LOG_PER_CLASS_METRICS"] = "true"
os.environ["COMET_MAX_IMAGE_UPLOADS"] = "0"

# initialize experiment in comet
comet_ml.init("ai-census") # it get the name of the project name on training

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(
                      data=DATASET_YAML, 
                      device = 0,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                      epochs = 200,
                      name = "2_exp_nobirds_gpu",      # experiment name
                      resume = RESUME,	            # resume training from last checkpoint
                      single_cls = True,	        # train multi-class data as single-class -> def = False
                      cfg="AI_Census/config/config.yaml",
                      )

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities'''

# Export the model to ONNX format
#success = model.export(format='onnx')