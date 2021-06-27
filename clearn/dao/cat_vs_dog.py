import numpy as np
import os
import cv2

from copy import deepcopy

from clearn.analysis import ImageConcept
from clearn.dao.concept_utils import get_label,generate_concepts_from_digit_image, get_concept_map
from clearn.dao.idao import IDao
import json
import pandas as pd
import glob
import fnmatch

target_image_shape = (256, 256)


def resize_images(data_dir):
    image_path = data_dir + "*.jpg"
    print(f"Loading images fromm {image_path}")
    num_images = len(fnmatch.filter(os.listdir(data_dir), '*.jpg') )
    images = np.zeros((num_images, target_image_shape[0], target_image_shape[1], 3))
    dst_path = data_dir.rsplit("/", 1)[0] + "/resized"
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    print("Resized output will be written to ", dst_path)
    for image_num, file in enumerate(glob.glob(image_path)):
        im = cv2.imread(file)
        im = cv2.resize(im, target_image_shape)
        dst_file = dst_path + "/" + file.rsplit("/", 1)[1]
        print("saving resized image to", dst_file)
        cv2.imwrite(dst_file, im)
        # print(im.shape)

    return images, None


class CatVsDogDao(IDao):
    MAP_FILE_NAME = "manually_generated_concepts.json"

    def __init__(self,
                 dataset_name: str,
                 split_name: str,
                 num_validation_samples: int,
                 dataset_path:str,
                 concept_id:int
                 ):
        self.dataset_name:str = "cat_vs_dog"
        self.dataset_path = dataset_path
        self.split_name:str = split_name
        self.num_validation_samples:int = num_validation_samples
        self.num_concepts_label_generated = 0
        self.concept_id = concept_id
        print(self.dataset_path, self.split_name, CatVsDogDao.MAP_FILE_NAME)
        map_filename = self.dataset_path + "/" + dataset_name+"/" + self.split_name + "/" + CatVsDogDao.MAP_FILE_NAME

        self.concepts_dict = None
        if os.path.isfile(map_filename):
            print(f"Reading concepts map from {map_filename}")
            self.concepts_dict = get_concept_map(map_filename)
        else:
            print(f"Map filename {map_filename}  does not exist. No concepts will be generated")

        self.label_key_to_label_map = dict()
        if self.concepts_dict is not None:
            for digit, list_of_concept_dict in self.concepts_dict.items():
                for image_concept_dict in list_of_concept_dict:
                    concept_image = ImageConcept.fromdict(image_concept_dict)
                    v_extend = concept_image.v_extend
                    h_extend = concept_image.h_extend
                    if len(v_extend) == 0:
                        v_extend = [0, 28]
                    if len(h_extend) == 0:
                        h_extend = [0, 28]
                    self.label_key_to_label_map[f"{digit}_{h_extend[0]}_{h_extend[1]}_{v_extend[0]}_{v_extend[1]}"] = self.label_key_to_label_map + self.num_concepts_label_generated
                    self.num_concepts_label_generated = self.num_concepts_label_generated + 1

        self.orig_train_images, self.orig_train_labels =  self.load_orig_train_images_and_labels(dataset_path + "/" +dataset_name)
        print(f"Images loaded  Image shape {self.orig_train_images}  Label shape {self.orig_train_labels.shape}")

        self.images_by_label = dict()
        for i in range(10):
            self.images_by_label[i] = self.orig_train_images[self.orig_train_labels == i]

        self.image_set_dict = dict()
        self.level2_manual_annotations_good_cluster = dict()
        level2_manual_annotations_good_cluster_filename = self.dataset_path + "/" + dataset_name+"/" + self.split_name + "/" + "level2_manual_annotations_good_cluster.json"

        if os.path.isfile(level2_manual_annotations_good_cluster_filename):
            with open(level2_manual_annotations_good_cluster_filename) as json_file:
                level2_manual_annotations_good_cluster = json.load(json_file)
            print([k for k in level2_manual_annotations_good_cluster.keys()])
            for cluster_id in range(10):
                self.image_set_dict[f"level_2_cluster_centers_{cluster_id}"] = np.asarray(
                    level2_manual_annotations_good_cluster[str(cluster_id)]["decoded_images"])
                self.image_set_dict[f"training_set_{cluster_id}"] = self.images_by_label[cluster_id]
        self.data_dict = None

    @property
    def number_of_training_samples(self):
        if self.data_dict is None:
            return self.orig_train_images.shape[0]
        else:
            return self.data_dict["TRAIN_INDICES"].shape[0]

    @property
    def num_concepts(self):
        return 0

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
        return 2

    @property
    def num_classes(self):
        return self.num_original_classes + self.num_concepts_label_generated

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
        tr_x, tr_y = self.load_train_images_and_label(data_dir, split_location + CatVsDogDao.MAP_FILE_NAME)
        if shuffle:
            seed = 547
            np.random.seed(seed)
            np.random.shuffle(tr_y)
            np.random.seed(seed)
            np.random.shuffle(tr_y)
        tr_y = tr_y.astype(int)
        y_vec = np.eye(self.num_classes)[tr_y]
        return tr_x / self.max_value, y_vec

    def load_orig_train_images_and_labels(self, data_dir):
        orig_train_images, orig_train_labels = resize_images(data_dir + "/train/")
        return orig_train_images, orig_train_labels

    def load_train_images_and_label(self, data_dir, map_filename=None):

        if map_filename is None:
            raise Exception("parameter map_filename can be None. Pass a valid path for loading concepts dictionary")
        if self.concept_id is None:
            raise Exception("Pass an integer for parameter concept_id while creating the instance of MnistConceptsDao")

        concept_image_filename = data_dir + f"concept_{self.concept_id}.csv"
        feature_dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]

        if os.path.isfile(concept_image_filename):
            concepts_df = pd.read_csv(concept_image_filename)
            x = concepts_df.values[:, 0:feature_dim]
            y = concepts_df.values[:, feature_dim]
            _x = x.reshape((x.shape[0], self.image_shape[0], self.image_shape[1], self.image_shape[2] ))

        else:
            concepts, concept_labels = self.generate_concepts(map_filename, num_images_per_concept=3000)
            # TODO verify the concepts once
            print(self.orig_train_images.shape, concepts.shape)
            _x = np.vstack([self.orig_train_images, concepts])
            print(self.orig_train_labels.shape, concept_labels.shape)
            y = np.hstack([self.orig_train_labels, concept_labels])
            print("After hstack and vstack", _x.shape, y.shape)
            x = deepcopy(_x).reshape(_x.shape[0], feature_dim)
            image_df = pd.DataFrame(x)
            image_df["label"] = y
            image_df.to_csv(concept_image_filename, index=False)
            print("Just before returning",_x.shape, x.shape)
        return _x, y

    def generate_concepts(self, map_filename, num_images_per_concept):
        concepts_dict = self.get_concept_map(map_filename)
        print(self.num_concepts_label_generated)
        # Change 8 to 12 below . 10 + 2 (4 and 9 has bimodal distribution)
        concepts = np.zeros((num_images_per_concept * self.num_concepts, 28, 28, 1))
        labels = np.zeros((num_images_per_concept * self.num_concepts), np.int8)
        num_concepts_generated = 0
        for digit, list_of_concept_dict in concepts_dict.items():
            concepts_for_digit, labels_for_concepts_for_digit = self.generate_concepts_for_digit(digit,
                                                                  list_of_concept_dict,
                                                                  num_images_per_concept,
                                                                          self.label_key_to_label_map
                                                                  )
            concepts[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = concepts_for_digit
            labels[num_concepts_generated: num_concepts_generated + concepts_for_digit.shape[0]] = labels_for_concepts_for_digit
            num_concepts_generated = num_concepts_generated + concepts_for_digit.shape[0]

        return concepts[0:num_concepts_generated], labels[0:num_concepts_generated]

    def get_samples(self, digit, cluster_name, sample_index):
        # TODO implement for multiple images
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
                image_for_concept = generate_concepts_from_digit_image(digit_image,
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

if __name__ == "__main__":
    dao = CatVsDogDao( dataset_name="cat_vs_dog",
                       dataset_path = "/Users/sunilv/concept_learning_exp/datasets",
                       split_name="split_70_30",
                       num_validation_samples=-1,
                       concept_id=1
                       )

    print(dao.orig_train_images.shape)
