import json
import cv2

# Cargar las categorías
categories = [
    {"id": 0, "name": "bird"},
    {"id": 1, "name": "cow"},
    {"id": 2, "name": "domestic dog"},
    {"id": 3, "name": "egyptian mongoose"},
    {"id": 4, "name": "european badger"},
    {"id": 5, "name": "european rabbit"},
    {"id": 6, "name": "fallow deer"},
    {"id": 7, "name": "genet"},
    {"id": 8, "name": "horse"},
    {"id": 9, "name": "human"},
    {"id": 10, "name": "iberian hare"},
    {"id": 11, "name": "iberian lynx"},
    {"id": 12, "name": "red deer"},
    {"id": 13, "name": "red fox"},
    {"id": 14, "name": "wild boar"},
]

# Cargar las imágenes desde el archivo de texto
with open('Dataset/test.txt', 'r') as file:
    image_paths = [line.strip() for line in file]

# Cargar los datos de las predicciones desde el archivo JSON
with open('runs/detect/val3/predictions.json', 'r') as file:
    predictions = json.load(file)

current_image_index = 0

while True:
    image_path = image_paths[current_image_index]
    image = cv2.imread(image_path)

    # Filtrar las predicciones para la imagen actual
    image_id = image_path.split('/')[-1]  # Obtener el ID de la imagen
    image_predictions = [pred for pred in predictions if pred['image_id'] == image_id]

    # Dibujar las bounding boxes en la imagen
    for pred in image_predictions:
        bbox = pred['bbox']
        category_id = pred['category_id']
        category = categories[category_id]['name']
        score = pred['score']
        
        color = (0, 255, 0)  # Color para la categoría (puedes asignar colores a las categorías)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)
        cv2.putText(image, f'{category} ({score:.2f})', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    '''
    # Mostrar la imagen con bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    
    '''
    key = cv2.waitKey(0)

    if key == ord('q'):  # Salir del bucle si se presiona 'q'
        break
    elif key == ord('a') and current_image_index > 0:  # Imagen anterior
        current_image_index -= 1
    elif key == ord('d') and current_image_index < len(image_paths) - 1:  # Imagen siguiente
        current_image_index += 1

cv2.destroyAllWindows()
