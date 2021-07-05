import numpy as np
import os
import gzip
import json
import pandas as pd

from copy import deepcopy
from clearn.analysis import ImageConcept
from clearn.dao.concept_utils import generate_concepts_from_digit_image, get_label, get_concept_map
from clearn.dao.idao import IDao
from clearn.concepts import OPERATORS, apply_operator

# MAP_FILE_NAME = "manually_generated_concepts.json"
MAP_FILE_NAME = "manually_generated_concepts_icvgip.json"

MAX_IMAGES_TO_DISPLAY = 12


def get_key(digit, h_extend, v_extend):
    return f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"


def get_params(key):
    p = key.split("_")
    return int(p[0]), (int(p[1]), int(p[2])), (int(p[3]), int(p[4]))


class MnistConceptsDao(IDao):

    NUM_IMAGES_PER_CONCEPT = 3000

    def __init__(self,
                 dataset_name: str,
                 split_name: str,
                 num_validation_samples: int,
                 dataset_path: str,
                 concept_id: int,
                 translate_image
                 ):
        self.translate_image = translate_image
        self.training_phase = "CONCEPTS"
        self.data_dict = None

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
            concepts, concept_labels, tops, lefts = self.generate_concepts(map_filename,
                                                                           MnistConceptsDao.NUM_IMAGES_PER_CONCEPT)

            print(self.orig_train_images.shape, concepts.shape)
            _x = np.vstack([self.orig_train_images, concepts])
            y = np.hstack([self.orig_train_labels, concept_labels])

            # Generate derived images
            concept_1, concept_2, derived_images, derived_labels = self.generate_derived_images(map_filename,
                                                                                                MnistConceptsDao.NUM_IMAGES_PER_CONCEPT)

            image_df = pd.DataFrame(concept_1, columns=["concept_1"])
            image_df["derived_label"] = derived_labels
            image_df["concept_2"] = concept_2
            image_df.to_csv(derived_images_filename, index=False)

            _x = np.vstack([_x, derived_images])
            y = np.hstack([y, derived_labels])

            x = deepcopy(_x).reshape(_x.shape[0], feature_dim)
            image_df = pd.DataFrame(x)
            image_df["label"] = y
            image_df["top"] = tops
            image_df["left"] = lefts

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
        # Change 8 to 12 below . 10 + 2 (4 and 9 has bimodal distribution)
        concepts = np.zeros((num_images_per_concept * self.num_concepts, 28, 28, 1))
        labels = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        tops = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        lefts = np.zeros((num_images_per_concept * self.num_concepts), np.int8)


        num_concepts_generated = 0
        for digit, list_of_concept_dict in concepts_dict.items():
            concepts_for_digit, labels_for_concepts_for_digit, tops_for_digit, lefts_for_digit = self.generate_concepts_for_digit(digit,
                                                                                                 list_of_concept_dict,
                                                                                                 num_images_per_concept,
                                                                                                 self.label_key_to_label_map
                                                                                                 )
            concepts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = concepts_for_digit
            tops[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = tops_for_digit
            lefts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = lefts_for_digit
            num_concepts_generated = num_concepts_generated + concepts_for_digit.shape[0]

        return concepts[0:num_concepts_generated], labels[0:num_concepts_generated], tops, lefts

    def get_samples(self, digit, cluster_name, sample_index):
        # TODO implement for multiple images
        return [self.image_set_dict[cluster_name][sample_index]]

    def generate_concepts_for_digit(self, digit, list_of_concept_dict, num_images_per_concept, label_key_to_label_map):
        concepts_for_digit = np.zeros((num_images_per_concept * len(list_of_concept_dict), 28, 28, 1))
        labels = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        tops = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)
        lefts = np.zeros(num_images_per_concept * len(list_of_concept_dict), np.int8)

        num_samples_generated = 0
        for image_concept_dict in list_of_concept_dict:
            concept_image = ImageConcept.fromdict(image_concept_dict)
            v_extend, h_extend = concept_image.v_extend, concept_image.h_extend
            # if len(v_extend) == 0:
            #     v_extend = [0, 28]
            # if len(h_extend) == 0:
            #     h_extend = [0, 28]

            # TODO modify this to get samples from multiple images
            digit_images = self.get_samples(digit, concept_image.cluster_name, concept_image.sample_index)
            num_images = len(digit_images)
            label = get_label(digit, h_extend, v_extend, label_key_to_label_map)
            for digit_image in digit_images:
                image_for_concept, tops_for_concept, lefts_for_concept = generate_concepts_from_digit_image(concept_image,
                                                                       digit_image,
                                                                       num_images_per_concept // num_images,
                                                                       translate_image=self.translate_image
                                                                       )

                concepts_for_digit[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = image_for_concept
                labels[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = label
                tops[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = tops_for_concept
                lefts[num_samples_generated:num_samples_generated + image_for_concept.shape[0]] = lefts_for_concept
                num_samples_generated += image_for_concept.shape[0]
        return concepts_for_digit[0:num_samples_generated], labels[0:num_samples_generated], tops, lefts


if __name__ == "__main__":
    dao = MnistConceptsDao(dataset_name="mnist_concepts",
                           dataset_path="C:/concept_learning_exp/datasets/",
                           split_name="split_70_30",
                           num_validation_samples=-1,
                           concept_id=1,
                           translate_image=False
                           )
    print(dao.label_key_to_label_map)
    images,labels = dao.load_train_images_and_label("C:\concept_learning_exp\datasets/",
                                    map_filename="C:\concept_learning_exp\datasets/mnist_concepts/split_70_30/manually_generated_concepts.json")

    # for k, v in dao.label_key_to_label_map.items():
    #     print(k, v)
    print(dao.orig_train_images.shape)
    print(images.shape, labels.shape)
