from ultralytics import YOLO
from constants import *
import os
import pandas as pd

# load best model from training results
best_model = YOLO(MODEL_WEIGHTS_BEST)


'''df = pd.read_csv(TEST_CSV)

imgs_list = df.transpose().values.tolist()

#new = list(np.concatenate(imgs_list))

# iterate through the sublist using List comprehension
flatList = [element for innerList in imgs_list for element in innerList]'''

# get predictions on best model
results = best_model.predict("Dataset/test/rev01/26", stream=True, save=True)

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list'''

# View results
for r in results:
    print(r.B)  # print the Probs object containing the detected class probabilities