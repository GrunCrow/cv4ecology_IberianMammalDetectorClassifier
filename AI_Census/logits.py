from predict_with_logits import *
from constants import *

model_path = get_best_model_weights("2_exp_batch_16_no_birds")

img_path = "Dataset/multispecies.jpeg"
#txt_path = "Dataset/validation.txt"
txt_path = "Dataset/old_network.txt"

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

# run inference
results = run_predict(input_path=txt_path, 
                      model_path=model_path, 
                      score_threshold=0.4, 
                      iou_threshold=0.5, 
                      save_image = False,
                      save_json=True,
                      category_mapping = category_mapping,
                      softmax_temperature_value = 3,
                      agnostic=True,
                      )

for result in results:
    print("\n")
    print("Bounding Box :" + str(result['bbox']))
    print("Logits :" + str(result['logits']))
    print("Activations :" + str(result['activations']))
    print("\n")

#plot_image(img_path, results)