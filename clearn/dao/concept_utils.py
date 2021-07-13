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
    if scale == 0:
        X = np.zeros(num_samples)
    else:
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

    ld = location_description(h_extend_mean, v_extend_mean, image_shape)
    location_key = f"{h_extend_mean[0]}_{h_extend_mean[1]}_{v_extend_mean[0]}_{v_extend_mean[1]}"

    digit_location_key = f"{digit}_{location_key}"
    segment_location_description = f"{ld} of {digit} "

    title = f"{segment_location_description}[{digit_location_key}]"
    title_for_filename = title.strip().replace(" ", "_")
    image_filename = None
    if path is not None:
        image_filename = path + f"seg_{title_for_filename}_{int(epochs_completed)}_{cluster}_{sample_index}.png"
    tops = np.zeros(len(v_extends), dtype=int)
    lefts = np.zeros(len(v_extends), dtype=int)
    image_number = 0
    for h_extend, v_extend in zip(h_extends, v_extends):
        cropped = image[ v_extend[0]:v_extend[0] + v_extend[1], h_extend[0]:h_extend[0] + h_extend[1]]
        h_im, _ = ImageConcept.tight_bound_h(cropped)
        cropped_and_stripped, _= ImageConcept.tight_bound_v(h_im)
        # if digit_location_key == "4_13_16_0_28":
        #     print(digit_location_key, np.sum(cropped_and_stripped), np.sum(cropped))
        # if np.sum(cropped_and_stripped) < 10:
        #     print(f"Skipping {digit_location_key} {np.sum(cropped_and_stripped)} ")
        #     continue

        if translate_image:
            tops[image_number] = randint(0, height - cropped_and_stripped.shape[0])
            lefts[image_number] = randint(0, width - cropped_and_stripped.shape[1])
            masked_images[image_number, tops[image_number]:tops[image_number] + cropped_and_stripped.shape[0], lefts[image_number]:lefts[image_number] + cropped_and_stripped.shape[1]] = cropped_and_stripped
        else:
            masked_images[image_number, v_extend[0]:v_extend[0] + v_extend[1], h_extend[0]:h_extend[0] + h_extend[1]] = cropped
        image_number += 1

    if display_image:
        display_images(masked_images[0:20],
                       image_filename=image_filename,
                       title=title,
                       num_images_to_display=num_images
                       )
    return masked_images[0:image_number], tops[0:image_number], lefts[0:image_number]


def get_label(digit,
              h_extend,
              v_extend,
              label_key_to_label_map):
    return label_key_to_label_map[f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"]


def generate_concepts_from_digit_image(concept_image:ImageConcept,
                                       digit_image,
                                       num_concepts_to_generate,
                                       path=None,
                                       translate_image=False,
                                       std_dev=1):

    cropped_and_stripped, h_extend, v_extend = concept_image.get_cropped_and_stripped()

    h_extends_from_random = normal_distribution_int(h_extend[0], std_dev, 3, num_concepts_to_generate)
    widths = normal_distribution_int(h_extend[1] - h_extend[0], std_dev, 3, num_concepts_to_generate)
    widths[widths == 0] = 1
    # width[width == 28] = 28
    v_extends_from_random = normal_distribution_int(v_extend[0], std_dev, 3, num_concepts_to_generate)
    heights = normal_distribution_int(v_extend[1] - v_extend[0], std_dev, 3, num_concepts_to_generate)
    heights[heights==0] = 1
    # v_extends_to_random = normal_distribution_int(v_extend[1], 1, 3, num_concepts_to_generate)

    concept_images, tops, lefts = segment_single_image_with_multiple_slices(digit_image,
                                                                            list(zip(h_extends_from_random, widths)),
                                                                            list(zip(v_extends_from_random, heights)),
                                                                            h_extend,
                                                                            v_extend,
                                                                            concept_image.digit,
                                                                            path,
                                                                            concept_image.cluster_name,
                                                                           concept_image.sample_index,
                                                                           display_image=False,
                                                                           epochs_completed=concept_image.epochs_completed,
                                                                           translate_image=translate_image
                                                                            )
    num_concepts_generated = concept_images.shape[0]
    widths = widths[0:num_concepts_generated]
    heights = heights[0:num_concepts_generated]
    print("Number of images generated", num_concepts_generated)
    return concept_images, tops, lefts, widths, heights

if __name__ == "__main__":
    digit = 7
    num_concepts_to_generate = 100
    map_filename="C:/concept_learning_exp/datasets/mnist_concepts/split_70_30/manually_generated_concepts_icvgip.json"
    concepts_dict = get_concept_map(map_filename)
    print([k for k in concepts_dict.keys()])
    concept_image = ImageConcept.fromdict(concepts_dict[str(3)][0])
    print(concept_image.h_extend, concept_image.v_extend)
    cropped_and_stripped, h_extend, v_extend = concept_image.get_cropped_and_stripped()
    # plt.imshow(cropped_and_stripped)
    # plt.show()
    print(h_extend, v_extend)

    h_extends_from_random = normal_distribution_int(h_extend[0], 1, 3, num_concepts_to_generate)
    widths = normal_distribution_int(h_extend[1] - h_extend[0], 1, 3, num_concepts_to_generate)
    widths[widths == 0] = 1
    v_extends_from_random = normal_distribution_int(v_extend[0], 1, 3, num_concepts_to_generate)
    heights = normal_distribution_int(v_extend[1] - v_extend[0], 1, 3, num_concepts_to_generate)
    heights[heights == 0] = 1

    concept_images, tops, lefts = segment_single_image_with_multiple_slices(concept_image.digit_image[0],
                                                                            list(zip(h_extends_from_random, widths)),
                                                                            list(zip(v_extends_from_random, heights)),
                                                                            h_extend,
                                                                            v_extend,
                                                                            concept_image.digit,
                                                                            None,
                                                                            concept_image.cluster_name,
                                                                            concept_image.sample_index,
                                                                            display_image=False,
                                                                            epochs_completed=concept_image.epochs_completed,
                                                                            translate_image=True
                                                                            )