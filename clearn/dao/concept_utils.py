import json
from random import randint

import numpy as np
from scipy.stats import truncnorm
import math

from matplotlib import pyplot as plt

from clearn.analysis import ImageConcept

MAP_FILE_NAME = "manually_generated_concepts.json"
MAX_IMAGES_TO_DISPLAY = 12


def get_concept_map(file_name):
    with open(file_name) as json_file:
        concept_map = json.load(json_file)
    return concept_map


def display_images(decoded_images,
                   image_filename,
                   title,
                   num_images_to_display=0,
                   fig_size=None,
                   axis=None,
                   num_cols=4,
                   ):

    colormap = "Greys"
    if fig_size is not None:
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
    else:
        fig = plt.figure()
    fig.tight_layout()
    num_images = decoded_images.shape[0]
    if num_images_to_display == 0:
        num_images_to_display = min(num_images, MAX_IMAGES_TO_DISPLAY)
    elif num_images_to_display > 0:
        num_images_to_display = min(num_images_to_display, MAX_IMAGES_TO_DISPLAY, num_images )
    else:
        raise Exception("num_images_to_display should not be negative")
    if num_images >  num_images_to_display:
        print(f"Number of image is {num_images}. Displaying only first {num_images_to_display} images ")
    num_rows = math.ceil(num_images_to_display / num_cols)
    if title is not None and len(title) > 0:
        fig.suptitle(title)
    for i in range(num_images_to_display):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.squeeze(decoded_images[i]), cmap=colormap)
        if axis is not None:
            ax.axis(axis)
    if image_filename is not None and len(image_filename) > 0:
        print(f"Saving the image to {image_filename}")
        plt.savefig(image_filename,
                    bbox="tight",
                    pad_inches=0
                    )
    plt.show()


def normal_distribution_int(mean, scale, range, num_samples):
    X = truncnorm(a=-range / scale, b=+range / scale, scale=scale).rvs(size=num_samples)
    X = X + mean
    X = X.round().astype(int)
    X[X < 0] = 0
    return X


def location_description(h_extend, v_extend, image_shape):
    # print(h_extend, v_extend, image_shape)
    # TODO  Convert interval into names
    h, w = image_shape[0], image_shape[1]
    # print(h_extend[0], w / 2., 0.2 * w)
    if abs(h_extend[0] - w / 2.) <= 0.2 * w:  # start from middle on horizontal axis
        # print("Starts from middle on horizontal axis")
        if v_extend[0] <= 0.1 * h:  # Starts from top on vertical axis
            # print("Starts from top on vertical axis")
            # print(v_extend[1], 0.9 * image_shape[0])
            if v_extend[1] >= 0.9 * image_shape[0]:  # Extend till end of vetrical axis
                return "right half"
    if abs(h_extend[0] - w / 2.) <= 0.2 * w:
        if v_extend[0] <= 0.1 * h and v_extend[1] >= 0.9 * h:
            return "left half"


def segment_single_image_with_multiple_slices(image,
                                              h_extends,
                                              v_extends,
                                              h_extend_mean,
                                              v_extend_mean,
                                              digit,
                                              path,
                                              cluster,
                                              sample_index,
                                              display_image=False,
                                              epochs_completed=0,
                                              translate_image=False
                                              ):

    height, width = image.shape[0], image.shape[1]
    image_shape = image.shape
    num_images = len(h_extends)

    masked_images = np.zeros([
        num_images,
        image.shape[0],
        image.shape[1],
        1
    ]
    )
    image_number = 0

    ld = location_description(h_extend_mean, v_extend_mean, image_shape)
    location_key = f"{h_extend_mean[0]}_{h_extend_mean[1]}_{v_extend_mean[0]}_{v_extend_mean[1]}"

    digit_location_key = f"{digit}_{location_key}"
    segment_location_description = f"{ld} of {digit} "

    title = f"{segment_location_description}[{digit_location_key}]"
    title_for_filename = title.strip().replace(" ", "_")
    image_filename = None
    if path is not None:
        image_filename = path + f"seg_{title_for_filename}_{int(epochs_completed)}_{cluster}_{sample_index}.png"

    for h_extend, v_extend in zip(h_extends, v_extends):
        if len(v_extend) == 0:
            v_extend = [0, height]
        if len(h_extend) == 0:
            h_extend = [0, width]
        cropped = image[ v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]]

        cropped_and_stripped = ImageConcept.tight_bound_h(ImageConcept.tight_bound_v(cropped))
        if translate_image:
            top = randint(0, image.shape[0] - cropped_and_stripped.shape[0])
            left = randint(0, image.shape[1] - cropped_and_stripped.shape[1])
            masked_images[image_number, top:top + cropped_and_stripped.shape[0], left:left + cropped_and_stripped.shape[1]] = cropped_and_stripped
        else:
            masked_images[image_number, v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]] = cropped

        image_number += 1

    if display_image:
        display_images(masked_images[0:20],
                       image_filename=image_filename,
                       title=title,
                       num_images_to_display=num_images
                       )
    return masked_images


def get_label(digit, h_extend, v_extend, label_key_to_label_map):
    return label_key_to_label_map[f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"]


def generate_concepts_from_digit_image(digit_image, num_concepts_to_generate, h_extend, v_extend, digit,
                                       cluster_name, sample_index, epochs_completed, path=None, translate_image=False):
    if len(v_extend) == 0:
        v_extend = [0, digit_image.shape[0]]
    if len(h_extend) == 0:
        h_extend = [0, digit_image.shape[1]]

    h_extends_from_random = normal_distribution_int(h_extend[0], 1, 3, num_concepts_to_generate)
    h_extends_to_random = normal_distribution_int(h_extend[1], 1, 3, num_concepts_to_generate)
    v_extends_from_random = normal_distribution_int(v_extend[0], 1, 3, num_concepts_to_generate)
    v_extends_to_random = normal_distribution_int(v_extend[1], 1, 3, num_concepts_to_generate)

    concept_images = segment_single_image_with_multiple_slices(digit_image,
                                                               list(zip(h_extends_from_random, h_extends_to_random)),
                                                               list(zip(v_extends_from_random, v_extends_to_random)),
                                                               h_extend,
                                                               v_extend,
                                                               digit,
                                                               path,
                                                               cluster_name,
                                                               sample_index,
                                                               display_image=False,
                                                               epochs_completed=epochs_completed,
                                                               translate_image=translate_image
                                                               )
    if digit == 4:
        print(h_extend, v_extend, np.sum(concept_images, axis=(1,2,3)).shape)

    return concept_images

