import numpy as np
import os
import gzip

from scipy.stats import truncnorm

from clearn.analysis import ImageConcept
from clearn.dao.idao import IDao
import json
import math

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

MAP_FILE_NAME = "manually_generated_concepts.json"
MAX_IMAGES_TO_DISPLAY = 12


def display_images(decoded_images,
                   image_filename,
                   title,
                   num_images_to_display = 0,
                   fig_size=None,
                   axis = None,
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
                                              epochs_completed=0
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
    title_for_filename = title.strip().replace(" " , "_")
    image_filename = None
    if path is not None:
        image_filename = path + f"seg_{title_for_filename}_{int(epochs_completed)}_{cluster}_{sample_index}.png"

    for h_extend, v_extend in zip(h_extends, v_extends):
        if len(v_extend) == 0:
            v_extend = [0, height]
        if len(h_extend) == 0:
            h_extend = [0, width]
        cropped = image[ v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]]
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


class MnistConceptsDao(IDao):
    def __init__(self,
                 dataset_name: str,
                 split_name: str,
                 num_validation_samples: int,
                 analysis_path,
                 dataset_path:str):
        self.dataset_name:str = "mnist_concepts"
        self.dataset_path = dataset_path
        self.split_name:str = split_name
        self.num_validation_samples:int = num_validation_samples
        self.num_concepts_label_generated = 0
        self.load_orig_train_images_and_labels(dataset_path+"mnist")
        self.image_set_dict = dict()
        self.level2_manual_annotations_good_cluster = dict()
        level2_manual_annotations_good_cluster_filename = "level2_manual_annotations_good_cluster.json"
        with open(analysis_path + "/" + level2_manual_annotations_good_cluster_filename) as json_file:
            level2_manual_annotations_good_cluster = json.load(json_file)
        print([k for k in level2_manual_annotations_good_cluster.keys()])
        for cluster_id in range(10):
            self.image_set_dict[f"level_2_cluster_centers_{cluster_id}"] = np.asarray(
                level2_manual_annotations_good_cluster[str(cluster_id)]["decoded_images"])
            self.image_set_dict[f"training_set_{cluster_id}"] = self.images_by_label[cluster_id]


    @property
    def number_of_training_samples(self):
        return 180000 - self.num_validation_samples

    @property
    def num_concepts(self):
        return 20

    @property
    def number_of_testing_samples(self):
        return 30000

    @property
    def image_shape(self):
        return [28, 28, 1]

    @property
    def max_value(self):
        return 255.

    @property
    def num_classes(self):
        return 10 + self.num_concepts_label_generated

    def load_test_1(self, data_dir):
        data_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(data_dir + 't10k-images-idx3-ubyte.gz',
                                 10000,
                                 16,
                                 28 * 28)
        x = data.reshape((10000, 28, 28, 1))
        data = self.extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        y = np.asarray(data.reshape(10000)).astype(np.int)
        return x, y

    def load_train_val(self,
                       data_dir,
                       shuffle=False,
                       stratified=None,
                       percentage_to_be_sampled=0.7,
                       split_location=None,
                       split_names=[],
                       seed=547):
        x, y = self.load_train_val_1(data_dir, split_location + MAP_FILE_NAME)
        _stratify = None
        if stratified:
            _stratify = y

        if len(split_names) == 2:
            splitted = train_test_split(x,
                                        y,
                                        test_size=percentage_to_be_sampled,
                                        stratify=_stratify,
                                        shuffle=shuffle,
                                        random_state=seed
                                        )
            train_x = splitted[0]
            val_x = splitted[1]
            train_y = splitted[2]
            val_y = splitted[3]
            dataset_dict = dict()
            dataset_dict["split_names"] = split_names
        else:
            raise Exception("Split not implemented for more than two splits")

        data_dict = self.create_data_dict(train_x, train_y, val_x, val_y)
        return data_dict

    def load_train(self, data_dir, shuffle, split_location=None):
        tr_x, tr_y = self.load_train_val_1(data_dir, split_location + MAP_FILE_NAME)
        if shuffle:
            seed = 547
            np.random.seed(seed)
            np.random.shuffle(tr_y)
            np.random.seed(seed)
            np.random.shuffle(tr_y)

        y_vec = np.eye(self.num_classes)[tr_y]
        return tr_x / self.max_value, y_vec

    def extract_data(self, filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            _data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return _data

    def load_orig_train_images_and_labels(self, data_dir):
        data_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(data_dir + 'train-images-idx3-ubyte.gz',
                                 60000,
                                 16,
                                 28 * 28)
        self.orig_train_images = data.reshape((60000, 28, 28, 1))
        data = self.extract_data(data_dir + '/train-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        self.orig_train_labels = np.asarray(data.reshape(60000)).astype(np.int)
        self.images_by_label = dict()
        for i in range(10):
            self.images_by_label[i] = self.orig_train_images[self.orig_train_labels == i]


    def load_train_val_1(self, data_dir, map_filename=None):
        if map_filename is None:
            raise Exception("parameter map_filename can be None. Pass a valid path for loading concepts dictionary")
        concepts, concept_labels = self.generate_concepts( map_filename, num_images_per_concept=6000)

        # TODO verify the concepts once
        print(self.orig_train_images.shape, concepts.shape)
        x = np.vstack([self.orig_train_images, concepts])
        print(x.shape)
        print(self.orig_train_labels.shape, concept_labels.shape)
        y = np.hstack([self.orig_train_labels, concept_labels])
        print(y.shape)
        return x, y

    def generate_concepts(self, map_filename, num_images_per_concept):
        concepts_dict = self.get_concept_map(map_filename)

        label_start = 10
        label_key_to_label_map = dict()
        current_label = label_start
        for digit, list_of_concept_dict in concepts_dict.items():
            for image_concept_dict in list_of_concept_dict:
                concept_image = ImageConcept.fromdict(image_concept_dict)
                v_extend = concept_image.v_extend
                h_extend = concept_image.h_extend
                if len(v_extend) == 0:
                    v_extend = [0, 28]
                if len(h_extend) == 0:
                    h_extend = [0, 28]

                label_key_to_label_map[f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"] = current_label
                current_label += 1
        self.num_concepts_label_generated = int(current_label)
        print(self.num_concepts_label_generated)
        # Change 8 to 12 below . 10 + 2 (4 and 9 has bimodal distribution)
        concepts = np.zeros((num_images_per_concept * self.num_concepts, 28, 28, 1))
        labels = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        num_concepts_generated = 0
        for digit, list_of_concept_dict in concepts_dict.items():
            concepts_for_digit, labels_for_concepts_for_digit = self.generate_concepts_for_digit(digit,
                                                                  list_of_concept_dict,
                                                                  num_images_per_concept,
                                                                          label_key_to_label_map
                                                                  )
            concepts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = concepts_for_digit
            labels[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = labels_for_concepts_for_digit
            num_concepts_generated = num_concepts_generated + concepts_for_digit.shape[0]

        return concepts[0:num_concepts_generated], labels[0:num_concepts_generated]

    def get_samples(self, digit, cluster_name, sample_index):
        # TODO implemnt for multiple images
        return [self.image_set_dict[cluster_name][sample_index]]

    def generate_concepts_for_digit(self, digit, list_of_concept_dict, num_images_per_concept, label_key_to_label_map):
        concepts_for_digit = np.zeros((num_images_per_concept * len(list_of_concept_dict), 28, 28, 1))
        labels = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        num_samples_generated = 0
        for image_concept_dict in list_of_concept_dict:
            concept_image = ImageConcept.fromdict(image_concept_dict)
            v_extend, h_extend = concept_image.v_extend, concept_image.h_extend
            if len(v_extend) == 0:
                v_extend = [0, 28]
            if len(h_extend) == 0:
                h_extend = [0, 28]

            # TODO modify this to get samples from multiple images
            digit_images = self.get_samples(digit, concept_image.cluster_name, concept_image.sample_index)
            num_images = len(digit_images)
            label = get_label(digit, h_extend, v_extend, label_key_to_label_map)
            for digit_image in digit_images:
                image_for_concept = self.generate_concepts_from_digit_image(digit_image,
                                                                            num_images_per_concept // num_images,
                                                                            h_extend,
                                                                            v_extend,
                                                                            digit,
                                                                            concept_image.cluster_name,
                                                                            concept_image.sample_index,
                                                                            concept_image.epochs_completed
                                                                            )

                concepts_for_digit[num_samples_generated:num_samples_generated+image_for_concept.shape[0]] = image_for_concept
                labels[num_samples_generated:num_samples_generated+image_for_concept.shape[0]] = label
                num_samples_generated += image_for_concept.shape[0]
        return concepts_for_digit[0:num_samples_generated], labels[0:num_samples_generated]

    def get_concept_map(self, file_name):
        with open(file_name) as json_file:
            concept_map = json.load(json_file)
        return concept_map

    def generate_concepts_from_digit_image(self, digit_image, num_concepts_to_generate, h_extend, v_extend, digit,
                                           cluster_name, sample_index, epochs_completed, path=None):
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
                                                                   epochs_completed=epochs_completed)
        return concept_images
