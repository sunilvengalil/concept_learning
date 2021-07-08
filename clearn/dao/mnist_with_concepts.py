import math

import cv2
import numpy as np
import os
import gzip
import json
import pandas as pd

from copy import deepcopy

import scipy.signal

from clearn.analysis import ImageConcept
from clearn.dao.concept_utils import generate_concepts_from_digit_image, get_label, get_concept_map, display_images
from clearn.dao.idao import IDao
from clearn.concepts import OPERATORS, apply_operator

# MAP_FILE_NAME = "manually_generated_concepts.json"
MAP_FILE_NAME = "manually_generated_concepts_icvgip.json"
MAX_IMAGES_TO_DISPLAY = 12
search_window_width = 4
search_window_height = 4
search_stride = 1


def get_key(digit, h_extend, v_extend):
    return f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"


def get_params(key):
    p = key.split("_")
    return int(p[0]), (int(p[1]), int(p[2])), (int(p[3]), int(p[4]))


def similarity_score(im_patch, cropped_and_stripped):
    # TODO implement for color image - correlation using lab color space
    im_patch = np.squeeze(im_patch)
    cropped_and_stripped = np.squeeze(cropped_and_stripped)
    if len(im_patch.shape) == 3:
        raise Exception("Similarity score not implemented for color images")
    return scipy.signal.correlate2d(im_patch, cropped_and_stripped)


def grid_search(digit_image, concept_image):
    cropped_and_stripped, h_extend, v_extend = concept_image.get_cropped_and_stripped()

    height, width = cropped_and_stripped.shape[0], cropped_and_stripped.shape[1]
    top_from = v_extend[0] - search_window_height
    top_to = v_extend[0] + search_window_height
    left_from = h_extend[0] - search_window_height
    left_to = h_extend[0] + search_window_height
    print(top_from, top_to, left_from, left_to)
    corr = scipy.signal.correlate2d(np.squeeze(digit_image), cropped_and_stripped, mode="same")

    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match


    # sim_scores = []
    # max_sim_score = 0
    # for top in range(top_from, top_to, search_stride):
    #     for left in range(left_from, left_to, search_stride) :
    #         im_patch = digit_image[top:top + height, left:left + width]
    #         print(im_patch.shape, cropped_and_stripped.shape)
    #         sim_score = similarity_score(im_patch, cropped_and_stripped)
    #         sim_scores.append(sim_score)
    #         print(sim_scores, sim_scores[-1])
    #         if sim_scores[-1] > max_sim_score:
    #             max_sim_score = sim_scores[-1]
    #             top_hat = top
    #             left_hat = left

    h_extend = [x - math.floor(cropped_and_stripped.shape[1] / 2.0), x + math.ceil(cropped_and_stripped.shape[1] / 2.0)]
    v_extend = [y - math.floor(cropped_and_stripped.shape[0] / 2.0), y + math.ceil(cropped_and_stripped.shape[0] / 2.0)]
    return h_extend, v_extend, corr


class MnistConceptsDao(IDao):
    NUM_IMAGES_PER_CONCEPT = 3000
    def __init__(self,
                 dataset_name: str,
                 split_name: str,
                 num_validation_samples: int,
                 dataset_path: str,
                 concept_id: int,
                 translate_image,
                 std_dev=1
                 ):
        self.translate_image = translate_image
        self.training_phase = "CONCEPTS"
        self.data_dict = None
        self.std_dev = std_dev

        self.dataset_name: str = "mnist_concepts"
        self.dataset_path = dataset_path
        self.split_name: str = split_name
        self.num_validation_samples: int = num_validation_samples
        self.num_concepts_label_generated = 0
        self.concept_id = concept_id
        print(self.dataset_path, self.split_name, MAP_FILE_NAME)
        map_filename = self.dataset_path + "/" + dataset_name + "/" + self.split_name + "/" + MAP_FILE_NAME
        print(f"Reading concepts map from {map_filename}")
        self.concepts_dict = get_concept_map(map_filename)

        label_start = self.num_original_classes
        self.label_key_to_label_map = dict()
        self.key_to_image_concept_map = dict()
        for digit, list_of_concept_dict in self.concepts_dict.items():
            for image_concept_dict in list_of_concept_dict:
                concept_image = ImageConcept.fromdict(image_concept_dict)
                v_extend = concept_image.v_extend
                h_extend = concept_image.h_extend
                if len(v_extend) == 0:
                    v_extend = [0, 28]
                if len(h_extend) == 0:
                    h_extend = [0, 28]

                key = get_key(digit, h_extend, v_extend)
                self.label_key_to_label_map[key] = label_start + self.num_concepts_label_generated
                self.key_to_image_concept_map[key] = concept_image
                self.num_concepts_label_generated = self.num_concepts_label_generated + 1

        self.key_to_label_map = dict()
        for key in self.label_key_to_label_map.keys():
            self.key_to_label_map[self.label_key_to_label_map[key]] = key

        self.num_derived_images_generated = len(OPERATORS)

        self.orig_train_images, self.orig_train_labels = self.load_orig_train_images_and_labels(dataset_path + "mnist")
        self.images_by_label = dict()
        for i in range(10):
            self.images_by_label[i] = self.orig_train_images[self.orig_train_labels == i]

        self.image_set_dict = dict()
        self.level2_manual_annotations_good_cluster = dict()
        level2_manual_annotations_good_cluster_filename = self.dataset_path + "/" + dataset_name + "/" + self.split_name + "/" + "level2_manual_annotations_good_cluster.json"
        with open(level2_manual_annotations_good_cluster_filename) as json_file:
            level2_manual_annotations_good_cluster = json.load(json_file)
        print([k for k in level2_manual_annotations_good_cluster.keys()])
        for cluster_id in range(10):
            self.image_set_dict[f"level_2_cluster_centers_{cluster_id}"] = np.asarray(
                level2_manual_annotations_good_cluster[str(cluster_id)]["decoded_images"])
            self.image_set_dict[f"training_set_{cluster_id}"] = self.images_by_label[cluster_id]

    @property
    def number_of_training_samples(self):
        if self.data_dict is None:
            return self.orig_train_images.shape[0]
        else:
            return self.data_dict["TRAIN_INDICES"].shape[0]

    @property
    def num_concepts(self):
        return self.num_concepts_label_generated

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
    def num_original_classes(self):
        return 10

    @property
    def num_classes(self):
        return self.num_original_classes + self.num_concepts_label_generated + self.num_derived_images_generated

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

    def load_train(self, data_dir, shuffle, split_location=None):
        if self.concept_id is None:
            raise Exception("Pass an integer for parameter concept_id while creating the MnistConceptsDao instance ")
        tr_x, tr_y = self.load_train_images_and_label(data_dir, split_location + MAP_FILE_NAME)
        if shuffle:
            seed = 547
            np.random.seed(seed)
            np.random.shuffle(tr_y)
            np.random.seed(seed)
            np.random.shuffle(tr_y)
        tr_y = tr_y.astype(int)
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
        orig_train_images = data.reshape((60000, 28, 28, 1))
        data = self.extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
        orig_train_labels = np.asarray(data.reshape(60000)).astype(np.int)
        return orig_train_images, orig_train_labels

    def load_train_images_and_label(self, data_dir, map_filename=None, training_phase=None):

        if map_filename is None:
            raise Exception("parameter map_filename can be None. Pass a valid path for loading concepts dictionary")
        if self.concept_id is None:
            raise Exception("Pass an integer for parameter concept_id while creating the instance of MnistConceptsDao")

        concept_image_filename = data_dir  + f"concept_{self.concept_id}.csv"
        derived_images_filename = data_dir + f"derived_{self.concept_id}.csv"

        feature_dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]

        if os.path.isfile(concept_image_filename):
            concepts_df = pd.read_csv(concept_image_filename)
            x = concepts_df.values[:, 0:feature_dim]
            y = concepts_df.values[:, feature_dim]
            _x = x.reshape((x.shape[0], self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        else:
            concepts, concept_labels, _tops, _lefts, _widths, _heights = self.generate_concepts(map_filename,
                                                                           MnistConceptsDao.NUM_IMAGES_PER_CONCEPT)
            print(np.unique(concept_labels,return_counts=True))
            print(self.orig_train_images.shape, concepts.shape)
            _x = np.vstack([self.orig_train_images, concepts])
            y = np.hstack([self.orig_train_labels, concept_labels])
            tops = np.hstack([np.zeros_like(self.orig_train_labels), _tops])
            lefts = np.hstack([np.zeros_like(self.orig_train_labels), _lefts])
            widths = np.hstack([np.zeros_like(self.orig_train_labels), _widths])
            heights = np.hstack([np.zeros_like(self.orig_train_labels), _heights])

            # Generate derived images
            concept_1, concept_2, derived_images, derived_labels = self.generate_derived_images(map_filename,
                                                                                                MnistConceptsDao.NUM_IMAGES_PER_CONCEPT)

            image_df = pd.DataFrame(concept_1, columns=["concept_1"])
            image_df["derived_label"] = derived_labels
            image_df["concept_2"] = concept_2
            image_df.to_csv(derived_images_filename, index=False)

            _x = np.vstack([_x, derived_images])
            y = np.hstack([y, derived_labels])
            tops = np.hstack([tops, np.zeros_like(derived_labels)])
            lefts = np.hstack([lefts, np.zeros_like(derived_labels)])
            widths = np.hstack([widths, np.zeros_like(derived_labels)])
            heights = np.hstack([heights, np.zeros_like(derived_labels)])

            x = deepcopy(_x).reshape(_x.shape[0], feature_dim)
            image_df = pd.DataFrame(x)
            image_df["label"] = y
            image_df["top"] = tops
            image_df["left"] = lefts
            image_df["widths"] = widths
            image_df["heights"] = heights
            image_df.to_csv(concept_image_filename, index=False)

        if training_phase == "CONCEPTS":
            _x = _x[(y >= self.num_original_classes) & (y < self.num_original_classes + self.num_concepts_label_generated)]
            y = y[(y >= self.num_original_classes) & (y < self.num_original_classes + self.num_concepts_label_generated)]
        return _x, y

    def generate_derived_images(self, map_filename, num_images_per_concept):
        concepts_dict = get_concept_map(map_filename)
        for k in concepts_dict.keys():
            print(k, [k1 for k1 in concepts_dict[k][0].keys()])
        operator_to_use = np.random.choice([0, 1],num_images_per_concept * len(OPERATORS))

        concepts_to_use_1, concepts_to_use_2, derived_images, derived_labels = apply_operator(operator_to_use,
                                                        self.key_to_image_concept_map,
                                                        self.key_to_label_map,
                                                        self.num_original_classes + self.num_concepts_label_generated)

        return concepts_to_use_1, concepts_to_use_2, derived_images, derived_labels

    def generate_concepts(self, map_filename, num_images_per_concept):
        concepts_dict = get_concept_map(map_filename)
        #  27 concepts (4 and 9 has bimodal distribution, mode 1 of 4 - 3 concepts, 8- 4 concepts)
        concepts = np.zeros((num_images_per_concept * self.num_concepts, 28, 28, 1))

        labels = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        tops = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        lefts = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        widths = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        heights = np.zeros((num_images_per_concept * self.num_concepts), np.int8)

        num_concepts_generated = 0
        for digit, list_of_concept_dict in concepts_dict.items():
            concepts_for_digit, labels_for_concepts_for_digit, tops_for_digit, lefts_for_digit, widths_for_digit, heights_for_digit = self.generate_concepts_for_digits(list_of_concept_dict,
                                                                                                                                                                        num_images_per_concept,
                                                                                                                                                                        self.label_key_to_label_map
                                                                                                                                                                        )
            concepts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = concepts_for_digit
            labels[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = labels_for_concepts_for_digit
            tops[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = tops_for_digit
            lefts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = lefts_for_digit
            widths[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = widths_for_digit
            heights[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = heights_for_digit

            num_concepts_generated = num_concepts_generated + concepts_for_digit.shape[0]


        return concepts[0:num_concepts_generated], labels[0:num_concepts_generated], tops[0:num_concepts_generated], lefts[0:num_concepts_generated], widths[0:num_concepts_generated], heights[0:num_concepts_generated]

    def get_samples(self,
                    digit,
                    num_samples_to_generate,
                    cluster_probabilities=(0.2, 0.8)
                    ) -> pd.DataFrame:
        cluster_num = 0
        sample_indices_for_cluster = [None] * len(self.image_set_dict)
        for k, v in self.image_set_dict.items():
            if k.endswith(str(digit)):
                print(cluster_probabilities)
                sample_indices_for_cluster[cluster_num] = np.random.choice(len(v),
                                                              num_samples_to_generate * min(cluster_probabilities[cluster_num] + 1, 1))
                cluster_num += 1

        cluster_indices = np.random.choice(len(cluster_probabilities), num_samples_to_generate, p=cluster_probabilities)
        print(cluster_indices.dtype)
        sample_indices = np.zeros(num_samples_to_generate, dtype=int)
        for cluster_num in range(len(cluster_probabilities)):
            num_samples_for_cluster = np.sum(cluster_indices == cluster_num)
            sample_indices[cluster_indices == cluster_num] = sample_indices_for_cluster[cluster_num][0:num_samples_for_cluster]

        samples = np.hstack([cluster_indices.reshape((num_samples_to_generate, 1)),
                             sample_indices.reshape(num_samples_to_generate, 1)])
        print(cluster_indices.shape, sample_indices.shape, samples.shape)
        df = pd.DataFrame(samples,columns=["cluster_indices", "sample_indices"])
        return df

    def generate_concepts_for_digits(self,
                                     list_of_concept_dict,
                                     num_images_per_concept,
                                     label_key_to_label_map,
                                     path=None):
        num_images_to_sample_from_for_digit = 100
        concepts_for_digit = np.zeros((num_images_per_concept * len(list_of_concept_dict), 28, 28, 1))
        labels = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        tops = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        lefts = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        widths = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        heights = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)

        num_samples_generated = 0
        for concept_no, image_concept_dict in enumerate(list_of_concept_dict):
            concept_image = ImageConcept.fromdict(image_concept_dict)

            v_extend, h_extend = concept_image.v_extend, concept_image.h_extend
            for digit in  concept_image.digits:
                clusters = []
                for k, v in self.image_set_dict.items():
                    if k.endswith(str(digit)):
                        clusters.append(v)

                sample_df = self.get_samples(digit,
                                             num_images_per_concept // (len(concept_image.digits) * num_images_to_sample_from_for_digit)
                                             )
                label = get_label(digit, h_extend, v_extend, label_key_to_label_map)
                print("Sample df shape",sample_df.shape)
                # TODO modify this using apply
                for index, sample_row in sample_df.iterrows():
                    cluster = clusters[sample_row["cluster_indices"]]
                    print(cluster.shape)
                    if path is not None:
                        display_images(cluster, path + f"cluster_{index}.png", f"cluster_{index}")
                    sample_index = sample_row["sample_indices"]
                    digit_image = cluster[sample_index]
                    print("Concept image: extends", concept_image.h_extend, concept_image.v_extend, concept_image.digit_image.shape, np.sum(concept_image.digit_image))
                    h_extend, v_extend, corr = grid_search(digit_image,
                                                           concept_image)
                    print(h_extend, v_extend, digit_image.shape)
                    plt.figure()
                    plt.imshow(np.squeeze(concept_image.digit_image))
                    print(concept_no, digit, index, sample_row["cluster_indices"], sample_index)
                    print("Digit image min and max", np.min(concept_image.digit_image), np.max(concept_image.digit_image))
                    print("Digit image from cluster min and max", np.min(digit_image), np.max(digit_image))

                    if path is not None:
                        cv2.imwrite(path + f"/correlation_{digit}_{concept_no}_{index}.png", corr)
                        cv2.imwrite(path + f"/original_{digit}_{concept_no}_{index}_{sample_index}.png", digit_image)

                        #cv2.imwrite(path + f"/concept_{digit}_{concept_no}_{index}_full.png", concept_image.get_full_image())
                        cv2.imwrite(path + f"/concept_{digit}_{concept_no}_{index}_full_digit.png", np.squeeze(concept_image.digit_image))
                        #cv2.imwrite(path + f"/concept_{digit}_{concept_no}_{index}_cropped.png", concept_image.get_cropped_image())

                    image_concept = ImageConcept(np.expand_dims(digit_image, 0),
                                                 h_extend,
                                                 v_extend,
                                                 digit,
                                                 num_clusters=concept_image.num_clusters,
                                                 cluster_name=concept_image.cluster_name,
                                                 sample_index=sample_index,
                                                 epochs_completed=concept_image.epochs_completed
                                                 )

                    image_for_concept, tops_for_concept, lefts_for_concept, widths_for_concepts, heights_for_concepts = generate_concepts_from_digit_image(image_concept,
                                                                                                                                                           digit_image,
                                                                                                                                                           num_images_per_concept//sample_df.shape[0],
                                                                                                                                                           translate_image=self.translate_image,
                                                                                                                                                           std_dev=self.std_dev
                                                                                                                                                           )
                    concepts_for_digit[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = image_for_concept
                    labels[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = label
                    tops[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = tops_for_concept
                    lefts[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = lefts_for_concept
                    widths[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = widths_for_concepts
                    heights[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = heights_for_concepts
                    num_samples_generated += image_for_concept.shape[0]
            plt.show()
        return concepts_for_digit[0:num_samples_generated], labels[0:num_samples_generated], tops[0:num_samples_generated], lefts[0:num_samples_generated], widths[0:num_samples_generated], heights[0:num_samples_generated]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    concept_id = 1
    dao = MnistConceptsDao(dataset_name="mnist_concepts",
                           dataset_path="C:/concept_learning_exp/datasets/",
                           split_name="split_70_30",
                           num_validation_samples=-1,
                           concept_id=concept_id,
                           translate_image=False
                           )
    print(dao.label_key_to_label_map)

    num_images_per_concept = 100
    path = f"C:/concept_learning_exp/datasets/mnist_concepts/concept_id_{concept_id}/"

    digit = 3
    concepts, labels, tops, lefts, widths, heights = dao.generate_concepts_for_digits(dao.concepts_dict[str(digit)],
                                                                                      num_images_per_concept=num_images_per_concept,
                                                                                      label_key_to_label_map=dao.label_key_to_label_map,
                                                                                      path=path
                                                                                      )
    print(f"Completed generating concepts for {digit}")
    print(concepts.shape)

    unique_labels, counts = np.unique(labels,return_counts=True)

    for label, count in zip(unique_labels, counts):
        concepts_for_label = concepts[labels == label]
        corr = np.zeros((concepts_for_label.shape[0],
                         concepts_for_label.shape[0]))

        for i in range(concepts_for_label.shape[0]):
            for j in range(concepts_for_label.shape[0]):
                corr[i, j] = np.dot(concepts_for_label[i].flatten(),
                                    concepts_for_label[j].flatten()) / (np.linalg.norm(concepts_for_label[i]) * np.linalg.norm(concepts_for_label[j]))
        print(corr.shape)
        print(np.sum(corr[corr > 0.5]))

        # plt.imshow(corr,cmap="gray")
        # plt.colorbar()
        # plt.show()

        top_indices = np.argsort(np.sum(corr, axis=0))
        print(top_indices)
        num_images_to_display = 32
        images_displayed = 0

        # for i in range(concepts_for_label.shape[0] // num_images_to_display):
        #     title = f"{i * num_images_to_display}_{(i + 1) * num_images_to_display}"
        #     fname = f"C:/concept_learning_exp/datasets/mnist_concepts/concept_id_{concept_id}/" + title
        #     print(concepts_for_label[top_indices[i * num_images_to_display : (i + 1) * num_images_to_display]].shape)
        #     display_images(concepts_for_label[top_indices[i * num_images_to_display : (i + 1) * num_images_to_display]],
        #                    fname,
        #                    title=title)
        #     # cv2.imwrite(f"C:/concept_learning_exp/datasets/mnist_concepts/concept_id_{concept_id}/concept_{label}_{i}.png",
        #     #             np.squeeze(concepts_for_label[i]))
        # if concepts_for_label.shape[0] % num_images_to_display > 0:
        #     left_out = (concepts_for_label.shape[0] // num_images_to_display) * num_images_to_display
        #     title = f"{left_out}_{concepts_for_label.shape[0]}"
        #     fname = f"C:/concept_learning_exp/datasets/mnist_concepts/concept_id_{concept_id}/" + title
        #     display_images(concepts_for_label[left_out: ],
        #                    fname,
        #                    title=title)


    # images,labels = dao.load_train_images_and_label("C:\concept_learning_exp\datasets/",
    #                                 map_filename="C:\concept_learning_exp\datasets/mnist_concepts/split_70_30/manually_generated_concepts.json")
    #
    # # for k, v in dao.label_key_to_label_map.items():
    # #     print(k, v)
    # print(dao.orig_train_images.shape)
    # print(images.shape, labels.shape)


