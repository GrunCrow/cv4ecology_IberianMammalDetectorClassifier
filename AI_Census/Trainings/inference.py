from ultralytics import YOLO
from constants import *
import os

# load best model from training results
best_model = YOLO(MODEL_WEIGHTS_BEST)

# get predictions on best model
results = best_model.predict("/content/Data/images/val/0a411d151f978818.png", save=True)

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities'''