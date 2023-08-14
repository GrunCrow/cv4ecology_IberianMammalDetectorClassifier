# First import useful packages
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# every time these files change, you need to restart the notebook kernel
# Hit the circle arrow Restart button in the toolbar
from metadata import Metadata 
import plotting

# NOTE: This script could take a few minutes to run depending on your computer's capacity. 
# Please be patient.
# If it does not terminate in 8 minutes, something is probably wrong.


def load_and_plot_boxes(metadata):
    '''Load detection results from the MegaDetector (precomputed for the competition)'''
    with open('metadata/iwildcam2022_mdv4_detections.json') as f:
        detections = json.load(f)
    plotting.plot_boxes_per_image(detections, metadata, save=True)


def find_len_empty_seq(metadata):
    print('-'*20)
    print(f'Length all sequences: {len(metadata.sequences)}')
    images_per_sequence = {seq:[] for seq in metadata.sequences}
    for im in metadata.images:
        images_per_sequence[im['seq_id']].append(im['id'])

    non_empty_sequences = []
    empty_sequences = []
    for seq in metadata.sequences:
        seq_categories = [metadata.im_to_cat[im] for im in images_per_sequence[seq]]
        if all([cat == 0 for cat in seq_categories]):
            empty_sequences.append(seq)
        else:
            non_empty_sequences.append(seq)
    print(f'Length empty sequences: {len(empty_sequences)}')
    print(f'Length non-empty sequences: {len(non_empty_sequences)}')


def part1():
    '''
    Prompt 1: Visualize statistics of your data

    Build useful visualizations of the statistics of your data. 
    - How many images or videos do you have? 
    - What distribution of categories of interest exist in your data? 
    - When was the data collected? 
    - Are there multiple types or modalities of data?'''
    # I'll be using the metadata from the 
    # iWildCam 2022 challenge (https://www.kaggle.com/c/iwildcam2022-fgvc9) as an example. 
    # We'll be looking at the training set only (since it has labels). 
    # It's pretty large (77G) so this will take a while...
    if not os.path.exists('iwildcam-2022-metadata.zip'):
        # NOTE: The ! in front of a command means it will be run in the shell
        os.system("curl -O https://lilablobssc.blob.core.windows.net/iwildcam/iwildcam-2022-metadata.zip")
        os.system("unzip iwildcam-2022-metadata.zip")

    # Let's load the metadata for the iWildCam 2022 training set and 
    # visualize some high-level statistics of the dataset.
    metadata = Metadata('metadata/iwildcam2022_train_annotations.json')
    metadata.show()

    plotting.plot_images_per_category(metadata, save=True)
    plotting.plot_images_per_category_per_location(metadata, save=True)
    plotting.plot_images_per_location(metadata, save=True)
    plotting.plot_categories_per_location(metadata, save=True)
    plotting.plot_images_per_sequence(metadata, save=True)

    find_len_empty_seq(metadata)
    load_and_plot_boxes(metadata)
   

def part2():
    '''
    Part2: load one of your images or videos and visualize it
    I can provide some starter code here if needed. 
    If you have annotations, you might also want to visualize the annotations!
    '''
    # TODO
    return


if __name__ == '__main__':
   part1()
   part2()
