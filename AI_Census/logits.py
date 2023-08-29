from predict_with_logits import *
from constants import *

model_path = get_best_model_weights("2_exp_batch_16_no_birds")

img_path = "Dataset/multispecies.jpeg"

# load the model
model, hooks = load_and_prepare_model(model_path)

# run inference
results = run_predict(img_path=img_path, 
                      model=model, 
                      hooks=hooks, 
                      threshold=0.5, 
                      iou=0.7, 
                      save_image = True)

for result in results:
    print("\n")
    print("Bounding Box :" + str(result['bbox']))
    print("Logits :" + str(result['logits']))
    print("Activations :" + str(result['activations']))
    print("\n")

#plot_image(img_path, results)