import comet_ml

from ultralytics import YOLO
from constants import *
from constants_unshared import MY_API_KEY_COMET
import os

os.environ["COMET_API_KEY"] = MY_API_KEY_COMET

# initialize experiment in comet
comet_ml.init("ai-census") # it get the name of the project name on training

# Create a new YOLO model from scratch
model = YOLO(MODEL_NAME)

# Load a pretrained YOLO model (recommended for training)
model = YOLO(MODEL_WEIGHTS)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=DATASET_YAML, 
                      epochs = 200,                   # number of epochs to train for
                      patience = 20,                # epochs to wait for no observable improvement for early stopping of training
                      batch = 128,                   # number of images per batch (-1 for AutoBatch)
                      save = True,                  # save train checkpoints and predict results
                      device = 1,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                      #workers = 8,                  # number of worker threads for data loading (per RANK if DDP)
                      project = "AI_Census/Trainings/YOLOv8",       # project name
                      name = "1_exp_batch_128",      # experiment name
                      # exist_ok = False,           # whether to overwrite existing experiment
                      pretrained = True,            # whether to use a pretrained model
                      optimizer = 'auto',           # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                      verbose = True,               # whether to print verbose output
                      seed = 42,	                # random seed for reproducibility
                      resume = RESUME,	            # resume training from last checkpoint
                      val = True,	                # validate/test during training 
                      save_json=True, 
                       
                      #single_cls = True,	        # train multi-class data as single-class

                      #cache = False,               # True/ram, disk or False. Use cache for data loading  
                      #deterministic = True,	    # whether to enable deterministic mode
                      #rect = False,	            # rectangular training with each batch collated for minimum padding
                      #cos_lr = False,	            # use cosine learning rate scheduler
                      #close_mosaic = 10,	        # (int) disable mosaic augmentation for final epochs (0 to disable)
                      #amp = True,	                # Automatic Mixed Precision (AMP) training, choices=[True, False]
                      #fraction = 1.0,	            # dataset fraction to train on (default is 1.0, all images in train set)
                      #profile = False,	            # profile ONNX and TensorRT speeds during training for loggers
                      #freeze = None,	            # (int or list, optional) freeze first n layers, or freeze list of layer indices during training
                      #lr0 = 0.01,	                # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                      #lrf = 0.01,	                # final learning rate (lr0 * lrf)
                      #momentum = 0.937,	        # SGD momentum/Adam beta1
                      #weight_decay = 0.0005,	    # optimizer weight decay 5e-4
                      #warmup_epochs = 3.0,	        # warmup epochs (fractions ok)
                      #warmup_momentum = 0.8,	    # warmup initial momentum
                      #warmup_bias_lr = 0.1,	    # warmup initial bias lr
                      #box = 7.5,	                # box loss gain
                      #cls = 0.5,	                # cls loss gain (scale with pixels)
                      #dfl = 1.5,	                # dfl loss gain
                      #label_smoothing = 0.0,	    # label smoothing (fraction)
                      #nbs = 64,	                # nominal batch size
                      

                      #dropout = 0.0,	            # use dropout regularization (classify train only) 
                      
                      #overlap_mask = True,	        # masks should overlap during training (segment train only)
                      #mask_ratio = 4,	            # mask downsample ratio (segment train only)
                      #pose = 12.0,	                # pose loss gain (pose-only)
                      #kobj = 2.0,	                # keypoint obj loss gain (pose-only)
                      )

# Evaluate the model's performance on the validation set
#results = model.val()


# load best model from training results
#best_model = YOLO('/content/YOLOv8-With-Comet/train/weights/best.pt')

# get predictions on best model
#results = best_model.predict("/content/Data/images/val/0a411d151f978818.png", save=True)

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# What Manuel wanted :´) -> for classification model, will it work with detection???????????????????
'''# Run inference on an image
results = model('bus.jpg')  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities'''

# Export the model to ONNX format
#success = model.export(format='onnx')