# PATHS
PATH = "AI_Census/"

# DATASET
# When running a jupyter notebook, it works like from the jupyter notebook location, 
# when runned from jupyter file, it runs from where user is at the terminal
DATASET_YAML = PATH + "aicensus.yaml"
TRAIN_TXT = PATH + "Data/TXTs/train_dataset_caltech.txt"
VAL_TXT = PATH + "Data/TXTs/validation_dataset_caltech.txt"
TEST_TXT = "Dataset/test.txt"
TEST_CSV = "Dataset/test.csv"

# MODEL
MODEL_NAME = 'yolov8s.yaml'

RESUME = False
MODEL_WEIGHTS_INITIAL = PATH + 'weights/yolov8s.pt' # created on path folder
MODEL_WEIGHTS_BEST = PATH + "Trainings/YOLOv8/2_exp_batch_16/weights/best.pt"

MODEL_WEIGHTS = MODEL_WEIGHTS_INITIAL # created on path folder

if MODEL_WEIGHTS == MODEL_WEIGHTS_BEST:
    RESUME = True

# ULTRALYTICS CODE

ULTRALYTICS_MULTI_LABEL = True  # Default Value = False -> to obtain all classes conf vector??