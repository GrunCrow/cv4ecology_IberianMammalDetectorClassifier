# PATHS
PATH = "ct_classifier/"

# DATASET
# When running a jupyter notebook, it works like from the jupyter notebook location, 
# when runned from jupyter file, it runs from where user is at the terminal
DATASET_YAML = PATH + "aicensus.yaml"
TRAIN_TXT = PATH + "Data/TXTs/train_dataset_caltech.txt"
VAL_TXT = PATH + "Data/TXTs/validation_dataset_caltech.txt"
TEST_TXT = PATH + "Data/TXTs/test_dataset_caltech.txt"

# MODEL
MODEL_NAME = 'yolov8n.yaml'
MODEL_WEIGHTS = PATH + 'weights/yolov8n.pt' # created on path folder