import comet_ml

from ultralytics import YOLO
from constants import *
#from constants_unshared import MY_API_KEY_COMET
#import os

#os.environ["COMET_API_KEY"] = MY_API_KEY_COMET

# initialize experiment in comet
#comet_ml.init("ai-census") # it get the name of the project name on training

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS_BEST)

metrics = model.val(#split="test", 
                    save_json = True,
                    save_hybrid = True,
                    device = 0,
                    plots = True,
                    #conf = 0.001
                    )