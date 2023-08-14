import os
import matplotlib.pyplot as plt
from metadata import Metadata

def plot_images_per_sequence(metadata, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    images_per_sequence = {seq: 0 for seq in metadata.sequences}
    for im in metadata.images:
        if im['id'] in metadata.im_to_cat:
            images_per_sequence[im['seq_id']] += 1

    sorted_sequences = sorted(metadata.sequences)

    plt.bar(range(len(sorted_sequences)), [images_per_sequence[seq] for seq in sorted_sequences], edgecolor='b')
    plt.xlabel('Sequence')
    plt.ylabel('Number of images')
    plt.title('Images per sequence')
    plt.xticks(range(len(sorted_sequences)), sorted_sequences, rotation='vertical')
    plt.grid(b=None)
    plt.tight_layout()

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/images_per_sequence.png')
    else:
        plt.show()



def plot_images_per_category_sorted(meta_data, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100
    images_per_category = {cat['id']: [] for cat in meta_data.categories}
    for im in meta_data.images:
        if im['id'] in meta_data.im_to_cat:
            images_per_category[meta_data.im_to_cat[im['id']]].append(im['id'])
    ind = range(len(meta_data.categories))
    sorted_cats = sorted(meta_data.categories, key=lambda cat: len(images_per_category[cat['id']]), reverse=True)
    plt.bar(ind,[len(images_per_category[cat['id']]) for cat in sorted_cats],edgecolor = 'b', log=True)
    plt.xlabel('Category')
    plt.ylabel('Number of images')
    plt.title('Images per category')
    plt.grid(b=None)
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.tick_params(axis='y', which='both', right=False, left=True)
    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/images_per_category.png')
    else:
        plt.show()


def plot_images_per_category_per_location(meta_data, save=False):
    num_locations = len(meta_data.locations)
    num_categories = len(meta_data.categories)

    # Determinar el tamaño de la figura en función del número de ubicaciones
    fig_width = 12
    fig_height = max(6, num_locations // 2)

    plt.clf()
    plt.rcParams['figure.figsize'] = [fig_width, fig_height]
    plt.rcParams['figure.dpi'] = 100

    images_per_category_per_loc = {loc: {cat['name']: 0 for cat in meta_data.categories} for loc in meta_data.locations}
    for im in meta_data.images:
        if im['id'] in meta_data.im_to_cat:
            category_id = meta_data.im_to_cat[im['id']]
            category_info = meta_data.get_category_by_id(category_id)
            images_per_category_per_loc[im['station']][category_info['name']] += 1

    ind = range(num_categories)
    sorted_cats = sorted(meta_data.categories, key=lambda cat: images_per_category_per_loc[meta_data.locations[0]][cat['name']], reverse=True)

    # Ordenar las ubicaciones de menor a mayor
    sorted_locations = sorted(meta_data.locations)

    # Definir la disposición de las subtramas
    num_cols = min(num_locations, 3)
    num_rows = (num_locations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

    for idx, loc in enumerate(sorted_locations):
        row_idx = idx // num_cols
        col_idx = idx % num_cols

        ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]

        ax.bar(ind, [images_per_category_per_loc[loc][cat['name']] for cat in sorted_cats], log=True)
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of images')
        ax.set_title('Location: ' + str(loc))
        ax.set_xticks([]) if idx < (num_rows - 1) * num_cols else ax.set_xticks(ind)
        ax.set_xticklabels([]) if idx < (num_rows - 1) * num_cols else ax.set_xticklabels([cat['name'] for cat in sorted_cats], rotation='vertical')
        ax.grid(b=None)

    plt.tight_layout()

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/images_per_category_per_location.png')
    else:
        plt.show()



def plot_images_per_location(meta_data, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    images_per_location = {loc: 0 for loc in meta_data.locations}
    for im in meta_data.images:
        if im['id'] in meta_data.im_to_cat:
            images_per_location[im['station']] += 1

    sorted_locations = sorted(meta_data.locations)

    plt.bar(range(len(sorted_locations)), [images_per_location[loc] for loc in sorted_locations], edgecolor='b', log=True)
    plt.xlabel('Location')
    plt.ylabel('Number of images')
    plt.title('Images per location')
    plt.xticks(range(len(sorted_locations)), sorted_locations, rotation='vertical')
    plt.grid(b=None)
    plt.tight_layout()

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/images_per_location.png')
    else:
        plt.show()


def plot_categories_per_location(meta_data, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    categories_per_location = {loc: set() for loc in meta_data.locations}
    for im in meta_data.images:
        if im['id'] in meta_data.im_to_cat:
            category_id = meta_data.im_to_cat[im['id']]
            category_info = meta_data.get_category_by_id(category_id)
            categories_per_location[im['station']].add(category_info['name'])

    sorted_locations = sorted(meta_data.locations)

    num_categories_per_location = [len(categories_per_location[loc]) for loc in sorted_locations]

    plt.bar(range(len(sorted_locations)), num_categories_per_location, edgecolor='b')
    plt.xlabel('Location')
    plt.ylabel('Number of categories')
    plt.title('Categories per location')
    plt.xticks(range(len(sorted_locations)), sorted_locations, rotation='vertical')
    plt.grid(b=None)
    plt.tight_layout()

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/categories_per_location.png')
    else:
        plt.show()


def plot_images_per_sequence(metadata, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    images_per_sequence = {seq:[] for seq in metadata.sequences}
    for im in metadata.images:
        images_per_sequence[im['seq_id']].append(im['id'])
    ind = range(len(metadata.sequences))
    plt.bar(ind,sorted([len(images_per_sequence[seq]) for seq in metadata.sequences],reverse=True),edgecolor = 'b', log=True)
    plt.xlabel('Sequence')
    plt.ylabel('Number of images')
    plt.title('Images per sequence')
    plt.grid(b=None)
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.tick_params(axis='y', which='both', right=False, left=True)
    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/images_per_sequence.png')
    else:
        plt.show()


def plot_boxes_per_image(detections, metadata, save=False):
    plt.clf()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 100

    # Count the number of boxes per image
    boxes_per_image = {}
    for ann in detections['annotations']:
        image_id = ann['image_id']
        if image_id not in boxes_per_image:
            boxes_per_image[image_id] = 1
        else:
            boxes_per_image[image_id] += 1

    # Count the number of images with a specific number of boxes
    num_boxes_counts = {}
    for num_boxes in boxes_per_image.values():
        if num_boxes not in num_boxes_counts:
            num_boxes_counts[num_boxes] = 1
        else:
            num_boxes_counts[num_boxes] += 1

    sorted_counts = sorted(num_boxes_counts.items())

    num_boxes, image_counts = zip(*sorted_counts)

    plt.bar(num_boxes, image_counts, edgecolor='b')
    plt.xlabel('Number of boxes')
    plt.ylabel('Number of images')
    plt.title('Number of boxes per image')
    plt.grid(b=None)
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.tick_params(axis='y', which='both', right=False, left=True)

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/boxes_per_image.png')
    else:
        plt.show()
