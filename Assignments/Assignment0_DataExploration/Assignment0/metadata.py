import json

EMPTY_NUMBER = 4

class Metadata:
    def __init__(self, filename):
        with open(filename) as f:
            metadata = json.load(f)

        # The metadata follows the COCO-CameraTraps data standard
        # https://github.com/microsoft/CameraTraps/blob/master/data_management/README.md
        
        self.images = metadata['images']
        self.annotations = metadata['annotations']
        self.train_categories = set([ann['category_id'] for ann in self.annotations])
        self.categories = [cat for cat in metadata['categories'] if cat['id'] in self.train_categories]
        self.locations = list(set(im['station'] for im in self.images))
        self.sequences = list(set(im['seq_id'] for im in self.images))
        self.im_to_cat = {ann['image_id']: ann['category_id'] for ann in self.annotations}

    def get_empty_images_count(self):
        all_image_ids = set(image['id'] for image in self.images)
        annotated_image_ids = set(annotation['image_id'] for annotation in self.annotations)
        empty_image_ids = all_image_ids - annotated_image_ids

        return len(empty_image_ids)

    def get_category_by_id(self, category_id):
        for category in self.categories:
            if category['id'] == category_id:
                return category
        return None

    def show(self):
        print('High-level statistics:\n')
        print('Images: '+str(len(self.images)))
        print('Categories: '+str(len(self.categories)))
        print('Annotations: '+str(len(self.annotations)))
        print('Animal images: ' + str(len(self.images) - self.get_empty_images_count()))
        print('Empty images: ' + str(self.get_empty_images_count()))
        print('Locations: '+str(len(self.locations)))
        print('Sequences: '+str(len(self.sequences)))