from predict_with_logits import *
from constants import *

model_path = get_best_model_weights("2_exp_batch_16_no_birds")

img_path = "Dataset/multispecies.jpeg"
txt_path = "Dataset/val_unique_locations.txt"

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

# load the model
model, hooks = load_and_prepare_model(model_path)

# run inference
results = run_predict(input_path=img_path, 
                      model=model, 
                      hooks=hooks, 
                      threshold=0.5, 
                      iou_threshold=0.7, 
                      save_image = True,
                      save_json=True,
                      category_mapping = category_mapping)

for result in results:
    print("\n")
    print("Bounding Box :" + str(result['bbox']))
    print("Logits :" + str(result['logits']))
    print("Activations :" + str(result['activations']))
    print("\n")

#plot_image(img_path, results)