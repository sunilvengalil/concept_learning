import os
import gzip
from copy import deepcopy

import numpy as np
import pandas as pd
import json
import imageio
from sklearn.model_selection import train_test_split

from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.dao.idao import IDao
from random import randint


def translate_random(im, max_pixels):
    im = im.reshape(28, 28)
    print(im.shape, max_pixels)
    num_pixels = randint(0, max_pixels)
    direction = randint(0, 3)
    if num_pixels > 0:
        if direction == 0:
            # move down
            im = np.pad(im, pad_width=((num_pixels,0), (0, 0)), mode="edge")[0:-num_pixels]
        elif direction == 1:
            # move up
            im = np.pad(im, pad_width=((0, num_pixels), (0, 0)), mode="edge")[num_pixels:]
        elif direction == 2:
            # move left
            im = np.pad(im, pad_width=( (0, 0), (0, num_pixels)), mode="edge")[:, num_pixels:]
        elif direction == 3:
            im = np.pad(im, pad_width=( (0, 0), (num_pixels, 0)), mode="edge")[:, 0:-num_pixels]
    im = im.reshape(1, 784)
    return im


def load_images(_config, train_val_data_iterator, dataset_type="train"):
    dao = _config.dao
    num_images = train_val_data_iterator.get_num_samples(dataset_type)
    feature_shape = list(train_val_data_iterator.get_feature_shape())
    num_images = (num_images // _config.BATCH_SIZE) * _config.BATCH_SIZE
    feature_shape.insert(0, num_images)
    train_images = np.zeros(feature_shape)
    i = 0
    train_labels = np.zeros([num_images, dao.num_classes])
    manual_annotations = np.zeros([num_images, dao.num_classes + 2])
    train_val_data_iterator.reset_counter(dataset_type)
    while train_val_data_iterator.has_next(dataset_type):
        batch_images, batch_labels, batch_annotations, _ = train_val_data_iterator.get_next_batch(dataset_type)
        if batch_images.shape[0] < _config.BATCH_SIZE:
            break
        train_images[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_images
        train_labels[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_labels
        manual_annotations[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_annotations
        i += 1
    return train_images, train_labels, manual_annotations


def generate_concepts(train_x_in, train_y, concept_map, percentage_concepts):
    train_y, test_y, indices_train, indices_test, train_x, test_x = train_test_split(train_y,
                                list(range(train_y.shape[0])),
                                train_x_in,
                                test_size=percentage_concepts,
                                stratify=train_y,
                                shuffle=True
                                )
    for i in test_y:
        v_extend = concept_map[i]["v_extend"]
        h_extend = concept_map[i]["h_extend"]

        cropped = train_x[i, v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]]
        masked_images = np.zeros([
            1,
            train_x.shape[1],
            train_x.shape[2],
            1
        ]
        )
        masked_images[:, v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]] = cropped

        train_x[i] = masked_images

    return train_x, train_y


class TrainValDataIterator:
    VALIDATION_Y_RAW = "validation_y_raw"
    VALIDATION_Y_ONE_HOT = "validation_y"
    VALIDATION_X = "validation_x"
    TRAIN_Y = "train_y"
    TRAIN_X = "train_x"
#    num_concepts = 20
    num_concepts_per_image_row: int
    num_concepts_per_image_col: int

    @classmethod
    def load_manual_annotation(cls, manual_annotation_file):
        df = pd.read_csv(manual_annotation_file)
        return df[["manual_annotation", "manual_annotation_confidence"]].values

    def load_train_val_existing_split(self, split_name, split_location):
        with open(split_location + split_name + ".json") as fp:
            dataset_dict = json.load(fp)

        split_names = dataset_dict["split_names"]
        for split in split_names:
            df = pd.read_csv(split_location + split + ".csv")
            dataset_dict[split] = df
        split_names = dataset_dict["split_names"]

        # TODO fix this remove hard coding of split name and column names
        train = dataset_dict["train"]
        x_columns = list(train.columns)
        x_columns.remove('label')
        data = train[x_columns].values
        train_x = data.reshape((data.shape[0],
                                self.dao.image_shape[0],
                                self.dao.image_shape[1],
                                self.dao.image_shape[2]))
        data = train[['label']].values
        train_y = np.asarray(data.reshape(data.shape[0])).astype(np.int32)

        val = dataset_dict["validation"]
        val_x = val[x_columns].values
        val_x = val_x.reshape(train_x.shape)

        val_y = val[['label']].values
        val_y = np.asarray(val_y.reshape(val_y.shape[0])).astype(np.int32)

        if len(split_names) != 2:
            raise Exception("Split not implemented for for than two splits")

        # TODO change this to numpy - remove the for loop performance improvement
        _val_y = np.zeros((len(val_y), self.dao.num_classes), dtype=np.float)
        for i, label in enumerate(val_y):
            _val_y[i, val_y[i]] = 1.0

        _train_y = np.zeros((len(train_y), self.dao.num_classes), dtype=np.float)
        for i, label in enumerate(train_y):
            _train_y[i, train_y[i]] = 1.0

        # TODO separate normalizing and loading logic
        return {TrainValDataIterator.TRAIN_X: train_x / self.dao.max_value,
                TrainValDataIterator.TRAIN_Y: _train_y,
                TrainValDataIterator.VALIDATION_X: val_x / self.dao.max_value,
                TrainValDataIterator.VALIDATION_Y_ONE_HOT: _val_y,
                TrainValDataIterator.VALIDATION_Y_RAW: val_y}

    @classmethod
    def from_existing_split(cls,
                            dao: IDao,
                            split_name: str,
                            split_location: str,
                            batch_size: int = 64,
                            manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                            manual_annotation_file: str = None,
                            budget=1,
                            manual_annotation_file_concepts=None,
                            num_concepts_per_image_row=7,
                            num_concepts_per_image_col=7,
                            translate_images=False,
                            percentage_concepts=0
                            ):
        """
        Creates and initialize an instance of TrainValDataIterator
        @param: split_name:Name of the train/valid/test split
        @param: split_location: path to folder where dataset split(train/val/test) is stored
        @param: batch_size: number of samples in each batch
        @param: manual_labels_config:
        """
        instance = cls(batch_size=batch_size, dao=dao)
        instance.translate_images = translate_images
        instance.max_pixels_to_translate = 4

        TrainValDataIterator.num_concepts_per_image_row = num_concepts_per_image_row
        TrainValDataIterator.num_concepts_per_image_col = num_concepts_per_image_col

        instance.concepts_gt = None
        # TODO convert this to lazy loading/use generator
        instance.dataset_dict = dao.load_train_val_existing_split(split_name, split_location)

        instance.train_x = instance.dataset_dict[TrainValDataIterator.TRAIN_X]
        instance.train_y = instance.dataset_dict[TrainValDataIterator.TRAIN_Y]
        instance.val_x = instance.dataset_dict[TrainValDataIterator.VALIDATION_X]
        instance.val_y = instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
        # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
        instance.unique_labels = np.unique(instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_RAW])
        instance.manual_labels_config = manual_labels_config
        _manual_annotation = None
        instance.budget = budget
        if manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
            if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                _manual_annotation = cls.load_manual_annotation(manual_annotation_file)
                print("Loaded manual annotation")
                print(f"Number of samples with manual confidence {sum(_manual_annotation[:, 1] > 0)}")
            else:
                # TODO if we are using random prior with uniform distribution, do we need to keep
                # manual confidence as 0.5 or 0
                print("Warning", "{} path does not exist. Creating random prior with uniform distribution".
                      format(manual_annotation_file))
                # create a numpy array of dimension (num_training_samples, num_unique_labels) and  set the one-hot encoded label
                # with uniform probability distribution for each label. i.e in case of MNIST each row will be set as one of the symbol
                # {0,1,2,3,4,5,6,7,8,9} with a probability of 0.1
                _manual_annotation = np.random.choice(dao.num_classes, len(instance.train_x))

        if manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
            # if manual_annotation_file_concepts is not None and os.path.isfile(manual_annotation_file_concepts):
            #     _manual_annotation_concepts = cls.load_manual_annotation(manual_annotation_file_concepts)
            #     print("Loaded manual annotation concepts")
            #     print(f"Number of samples with manual confidence {sum(_manual_annotation_concepts[:, 1] > 0)}")
            # else:
            #     # TODO if we are using random prior with uniform distribution, do we need to keep
            #     # manual confidence as 0.5 or 0
            #     print("Warning", "{} path does not exist. Creating random prior with uniform distribution".
            #           format(manual_annotation_file_concepts))
            #     # create a numpy array of dimension (num_training_samples, num_unique_labels) and  set the one-hot encoded label
            #     # with uniform probability distribution for each label. i.e in case of MNIST each row will be set as one of the symbol
            #     # {0,1,2,3,4,5,6,7,8,9} with a probability of 0.1
            #     _manual_annotation_concepts = np.random.choice(instance.dao.num_classes,
            #                                                    size=(len(instance.train_x),
            #                                                          TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col)
            #                                                    )

            _manual_annotation_concepts = np.zeros((len(instance.train_x), 13))


        instance.manual_annotation = instance.get_manual_annotation(manual_annotation_file,
                                                                    _manual_annotation,
                                                                    dao.num_classes,
                                                                    instance.train_y)

        instance.manual_annotation_concepts = instance.get_manual_annotation_concepts(manual_annotation_file_concepts,
                                                                                      _manual_annotation_concepts,
                                                                                      instance.dao.num_classes,
                                                                                      instance.concepts_gt
                                                                                      )

        print(f"Total Manual annotation confidence {np.sum(instance.manual_annotation[:, 10])}")

        if percentage_concepts > 0:
            instance.train_x_orig = deepcopy(instance.train_x)
            instance.train_x = generate_concepts(instance.train_x_orig, instance.train_y, percentage_concepts)
        if instance.translate_image:
            translated = np.apply_along_axis(translate_random, 1,
                                             instance.train_x.reshape(instance.train_x.shape[0], 784),
                                             max_pixels=instance.max_pixels_to_translate)
            instance.train_x = translated.reshape((instance.train_x.shape[0], 28, 28, 1))

        instance.train_idx = 0
        instance.val_idx = 0
        return instance

    def get_manual_annotation_concepts(self, manual_annotation_file, _manual_annotation, num_labels, actual_labels):
        if self.manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
            manual_annotation = np.zeros((len(_manual_annotation),
                                          TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col,
                                          num_labels + 1), dtype=np.float)
            if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                for i in range(len(_manual_annotation)):
                    for j in range(
                            TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col):
                        manual_annotation[i, j, int(_manual_annotation[i, j])] = 1.0
                        manual_annotation[i, j, num_labels] = _manual_annotation[
                            i, TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col + j]
            else:
                for i, label in enumerate(_manual_annotation):
                    for j in range(
                            TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col):
                        manual_annotation[i, j, _manual_annotation[i, j]] = 1.0
                        manual_annotation[i, j, num_labels] = 0  # set manual annotation confidence as 0
        elif self.manual_labels_config == ExperimentConfig.USE_ACTUAL:
            if actual_labels is not None and actual_labels.shape[0] == len(self.train_x):
                manual_annotation = np.zeros((len(self.train_x),
                                              TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col,
                                              num_labels + 1), dtype=np.float)
                manual_annotation[:, :, 0:num_labels] = actual_labels
                manual_annotation[:, :, num_labels] = 1  # set manual annotation confidence as 1
            else:
                raise Exception("Grount truth not set")
        return manual_annotation

    def get_manual_annotation(self, manual_annotation_file, _manual_annotation, num_labels, actual_labels):
        if self.manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
            manual_annotation = np.zeros((len(_manual_annotation), num_labels + 2), dtype=np.float)
            if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                for i, label in enumerate(_manual_annotation):
                    manual_annotation[i, int(_manual_annotation[i, 0])] = 1.0
                    manual_annotation[i, num_labels] = _manual_annotation[i, 1]
                    manual_annotation[i, num_labels+1] = 3
            else:
                for i, label in enumerate(_manual_annotation):
                    manual_annotation[i, _manual_annotation[i]] = 1.0
                    manual_annotation[i, num_labels] = 0  # set manual annotation confidence as 0
        elif self.manual_labels_config == ExperimentConfig.USE_ACTUAL:
            if actual_labels is not None and actual_labels.shape[0] == len(self.trai_xn):
                manual_annotation = np.zeros((len(self.train_x), num_labels + 1), dtype=np.float)
                if self.budget < 1:
                    indices = np.random.choice(len(self.train_x), int(self.budget * len(self.train_x)), replace=False)
                    print(f"Using labels of {len(indices)} samples")
                    manual_annotation[indices, 0:num_labels] = actual_labels[indices]
                    manual_annotation[indices, num_labels] = 1  # set manual annotation confidence as 1
                else:
                    manual_annotation[:, 0:num_labels] = actual_labels
                    manual_annotation[:, num_labels] = 1  # set manual annotation confidence as 1
            else:
                raise Exception("Grount truth not set")
        return manual_annotation

    def __init__(self,
                 dao: IDao,
                 dataset_path=None,
                 shuffle=False,
                 stratified=None,
                 validation_samples=128,
                 split_location=None,
                 split_names=[],
                 batch_size=None,
                 manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                 manual_annotation_file=None,
                 budget=1,
                 seed=547,
                 manual_annotation_file_concepts=None,
                 num_concepts_per_image_row=7,
                 num_concepts_per_image_col=7,
                 translate_image=False
                 ):
        TrainValDataIterator.num_concepts_per_image_row = num_concepts_per_image_row
        TrainValDataIterator.num_concepts_per_image_col = num_concepts_per_image_col
        self.translate_image = translate_image
        self.budget = budget
        self.train_idx = 0
        self.val_idx = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dao = dao
        if dataset_path is not None:
            self.entire_data_x, self.entire_data_y = load_train(dataset_path, dao=dao, split_location=split_location)
            if validation_samples == -1:
                percentage_to_be_sampled = 0.3
            else:
                percentage_to_be_sampled = validation_samples / len(self.entire_data_y)

            self.dataset_dict = load_train_val(dataset_path,
                                               shuffle=shuffle,
                                               stratified=stratified,
                                               percentage_to_be_sampled=percentage_to_be_sampled,
                                               split_location=split_location,
                                               split_names=split_names,
                                               dao=dao,
                                               seed=seed
                                               )
            self.train_x = self.dataset_dict[TrainValDataIterator.TRAIN_X]
            self.train_y = self.dataset_dict[TrainValDataIterator.TRAIN_Y]
            self.val_x = self.dataset_dict[TrainValDataIterator.VALIDATION_X]
            self.val_y = self.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
            self.unique_labels = np.unique(self.dataset_dict["validation_y_raw"])
            self.manual_labels_config = manual_labels_config
            self.concepts_gt = None
            _manual_annotation = None
            if manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
                if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                    _manual_annotation_all = TrainValDataIterator.load_manual_annotation(manual_annotation_file)
                    # fname = manual_annotation_file.rsplit("/", 1)[1]
                    # print(fname)
                    # manual_annotation_file_val = manual_annotation_file.rsplit("/", 1)[0] + "/" + fname.rsplit(".", 1)[0] +"_val" + ".csv"
                    # if os.path.isfile(manual_annotation_file_val):
                    #     raise Exception(f"File does not exist {manual_annotation_file_val}")
                    print(f"Loaded manual annotation. Shape:{_manual_annotation_all.shape}")
                    _manual_annotation = _manual_annotation_all[self.dataset_dict["TRAIN_INDICES"]]
                    _manual_annotation_val = _manual_annotation_all[self.dataset_dict["VAL_INDICES"]]
                    print(f"Number of samples with manual confidence {sum(_manual_annotation[:, 1] > 0)}")
                else:
                    # TODO if we are using random prior with uniform distribution, do we need to keep
                    # manual confidence as 0.5 or 0
                    print("Warning", "{} path does not exist. Creating random prior with uniform distribution".
                          format(manual_annotation_file))
                    """
                    Create a numpy array of dimension (num_training_samples, num_unique_labels) and  set the one-hot encoded label with uniform probability distribution for each label.
                    In case of MNIST each row will be set as one of the symbol {0,1,2,3,4,5,6,7,8,9} with a probability of 0.1
                    """
                    _manual_annotation = np.random.choice(self.unique_labels,
                                                          len(self.train_x))
                    _manual_annotation_val = np.random.choice(self.unique_labels,
                                                          len(self.val_x))

            _manual_annotation_concepts = None
            if manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
                if manual_annotation_file_concepts is not None and os.path.isfile(manual_annotation_file_concepts):
                    _manual_annotation_concepts = TrainValDataIterator.load_manual_annotation(
                        manual_annotation_file_concepts)
                    print("Loaded manual annotation concepts")
                    print(f"Number of samples with manual confidence {sum(_manual_annotation_concepts[:, 1] > 0)}")
                else:
                    # TODO if we are using random prior with uniform distribution, do we need to keep
                    # manual confidence as 0.5 or 0
                    print("Warning", "{} path does not exist. Creating random prior with uniform distribution".
                          format(manual_annotation_file))
                    """
                    Create a numpy array of dimension (num_training_samples, num_unique_labels) and  set the one-hot encoded label with uniform probability distribution for each label.
                    In case of MNIST each row will be set as one of the symbol {0,1,2,3,4,5,6,7,8,9} with a probability of 0.1
                    """
                    _manual_annotation_concepts = np.random.choice(dao.num_classes, size=(len(self.train_x),
                                                                                          TrainValDataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col)
                                                                   )

            self.manual_annotation = self.get_manual_annotation(manual_annotation_file,
                                                                _manual_annotation,
                                                                dao.num_classes,
                                                                self.train_y)
            self.val_manual_annotation = self.get_manual_annotation(manual_annotation_file,
                                                                _manual_annotation_val,
                                                                dao.num_classes,
                                                                self.val_y)


            self.manual_annotation_concepts = self.get_manual_annotation_concepts(manual_annotation_file_concepts,
                                                                                  _manual_annotation_concepts,
                                                                                  self.dao.num_classes,
                                                                                  self.concepts_gt
                                                                                  )
            self.max_pixels_to_translate = 4
            if translate_image:
                translated = np.apply_along_axis(translate_random, 1, self.train_x.reshape(self.train_x.shape[0], 784),
                                                 max_pixels=self.max_pixels_to_translate)
                self.train_x = translated.reshape((self.train_x.shape[0], 28, 28, 1))
            self.train_idx = 0
            self.val_idx = 0

    def has_next_val(self):
        # TODO fix this to handle last batch
        return self.val_idx * self.batch_size < self.val_x.shape[0]

    def has_next_train(self):
        # TODO fix this to handle last batch
        return self.train_idx * self.batch_size - self.train_x.shape[0] < self.batch_size

    def has_next(self, dataset_type):
        # TODO fix this to handle last batch
        if dataset_type == "train":
            return self.train_idx * self.batch_size - self.train_x.shape[0] < self.batch_size
        elif dataset_type == "val":
            return self.val_idx * self.batch_size < self.val_x.shape[0]
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")

    def get_feature_shape(self):
        return self.val_x.shape[1:]

    def get_next_batch(self, dataset_type):
        if self.batch_size is None:
            raise Exception("batch_size attribute is not set")
        if dataset_type == "train":
            x = self.train_x[self.train_idx * self.batch_size:(self.train_idx + 1) * self.batch_size]
            y = self.train_y[self.train_idx * self.batch_size:(self.train_idx + 1) * self.batch_size]
            label = self.manual_annotation[self.train_idx * self.batch_size:(self.train_idx + 1) * self.batch_size]
            label_concepts = self.manual_annotation_concepts[
                             self.train_idx * self.batch_size:(self.train_idx + 1) * self.batch_size]
            self.train_idx += 1
        elif dataset_type == "val":
            x = self.val_x[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            y = self.val_y[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            label = self.val_manual_annotation[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            label_concepts = self.manual_annotation_concepts[
                             self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            # TODO check if this is last batch, if yes,reset the counter
            self.val_idx += 1
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")
        return x, y, label, label_concepts

    def get_num_samples(self, dataset_type):
        if dataset_type == "train":
            return len(self.train_x)
        elif dataset_type == "val":
            return len(self.val_x)
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")

    def reset_counter(self, dataset_type):
        if self.translate_image:
            translated = np.apply_along_axis(translate_random, 1, self.train_x.reshape(self.train_x.shape[0], 784),
                                             max_pixels=self.max_pixels_to_translate)
            self.train_x = translated.reshape((self.train_x.shape[0], 28, 28, 1))
        if dataset_type == "train":
            self.train_idx = 0
        elif dataset_type == "val":
            self.val_idx = 0
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")

    def get_unique_labels(self):
        if self.unique_labels is None:
            # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
            self.unique_labels = np.unique(self.val_y)
        return self.unique_labels

    def save_val_images(self, dataset_path):
        def inverse_transform(images):
            return (images + 1.) / 2.

        def merge(images, size):
            print(images.shape)
            h, w = images.shape[1], images.shape[2]
            if (images.shape[3] in (3, 4)):
                c = images.shape[3]
                img = np.zeros((h * size[0], w * size[1], c))
                for idx, image in enumerate(images):
                    i = idx % size[1]
                    j = idx // size[1]
                    img[j * h:j * h + h, i * w:i * w + w, :] = image
                return img
            elif images.shape[3] == 1:
                img = np.zeros((h * size[0], w * size[1]))
                for idx, image in enumerate(images):
                    i = idx % size[1]
                    j = idx // size[1]
                    img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
                return img
            else:
                raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

        def imsave(images, size, path):
            image = np.squeeze(merge(images, size))
            imageio.imwrite(path, image)

        self.reset_counter("val")
        batch_no = 0
        manifold_w = 4
        manifold_h = self.batch_size // manifold_w
        while self.has_next("val"):
            val_images, _, _ = self.get_next_batch("val")
            file = "im_" + str(batch_no) + ".png"
            imsave(inverse_transform(val_images), [manifold_h, manifold_w], dataset_path + file)
            # save_image(val_images, [manifold_h, manifold_w], dataset_path + file)
            batch_no += 1
        self.reset_counter("val")


def load_train(data_dir,
               dao: IDao,
               shuffle=True,
               split_location=None):
    return dao.load_train(data_dir, shuffle, split_location)


def load_test_raw_data(data_dir):
    data_dir = os.path.join(data_dir, "images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            _data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return _data

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    test_x = data.reshape((10000, 28, 28, 1))
    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    test_y = data.reshape(10000)
    return test_x, test_y


def load_train_val(data_dir,
                   dao: IDao,
                   shuffle=False,
                   stratified=None,
                   percentage_to_be_sampled=0.7,
                   split_location=None,
                   split_names=[],
                   seed=547):

    return dao.load_train_val(data_dir,
                              shuffle,
                              stratified,
                              percentage_to_be_sampled,
                              split_location,
                              split_names,
                              seed=seed)


def load_test(data_dir,
              dao: IDao
              ):
    return dao.load_test(data_dir)


class DataIterator:
    Y_RAW = "test_y"
    Y_ONE_HOT = "test_y_one_hot"
    X = "test_x"
    #num_concepts = 20

    def __init__(self,
                 dao: IDao,
                 dataset_path=None,
                 batch_size=None,
                 num_concepts_per_image_row=7,
                 num_concepts_per_image_col=7
                 ):
        DataIterator.num_concepts_per_image_row = num_concepts_per_image_row
        DataIterator.num_concepts_per_image_col = num_concepts_per_image_col

        self.idx = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dao = dao
        if dataset_path is not None:
            self.dataset_dict = load_test(data_dir=dataset_path,
                                          dao=dao
                                          )
            self.x = self.dataset_dict[DataIterator.X]
            self.y = self.dataset_dict[DataIterator.Y_ONE_HOT]
            self.unique_labels = np.unique(self.dataset_dict[DataIterator.Y_RAW])

        self.manual_annotation = np.zeros((len(self.x), dao.num_classes + 2), dtype=np.float16)
        self.manual_annotation[:, 0:dao.num_classes] = self.y
        self.manual_annotation[:, dao.num_classes] = 1  # set manual annotation confidence as 1
        self.manual_annotation[:, dao.num_classes + 1] = -1

        self.manual_annotation_concepts = np.zeros((len(self.x),
                                                    DataIterator.num_concepts_per_image_row * TrainValDataIterator.num_concepts_per_image_col,
                                                    dao.num_classes + 1),
                                                   dtype=np.float)
        self.idx = 0

    def has_next(self, dataset_type):
        # TODO fix this to handle last batch
        return self.idx * self.batch_size - self.x.shape[0] < self.batch_size

    def get_feature_shape(self):
        return self.x.shape[1:]

    def get_next_batch(self, dataset_type):
        if self.batch_size is None:
            raise Exception("batch_size attribute is not set")
        x = self.x[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        y = self.y[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        label = self.manual_annotation[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        label_concepts = self.manual_annotation_concepts[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]

        self.idx += 1
        return x, y, label, label_concepts

    def get_num_samples(self, dataset_type):
        return len(self.x)

    def reset_counter(self, dataset_type):
        self.idx = 0

    def get_unique_labels(self):
        if self.unique_labels is None:
            self.unique_labels = np.unique(self.y)
        return self.unique_labels


if __name__ == "__main__":
    # Test cases for load_images

    dataset_path_1 = "/Users/sunilv/concept_learning_exp/datasets/cifar_10/"
    split_location_1 = dataset_path_1 + "test/"
    cifar_10_dao = get_dao("cifar_10", "test", 128)
    # DataIterator = DataIterator(dataset_path=dataset_path,
    #                             split_location=split_location,
    #                             split_names=["test"],
    #                             batch_size=128,
    #                             dao=cifar_10_dao)

    data_iterator = DataIterator.from_existing_split("test",
                                                     split_location=split_location_1,
                                                     dao=cifar_10_dao)
    while data_iterator.has_next("test"):
        x, y, _ = data_iterator.get_next_batch("test")
        print(x.shape)
        print(y.shape)

    print("completed_iteration")
    # N_3 = 16
    # N_2 = 128
    # Z_DIM = 20
    # run_id = 1
    #
    # ROOT_PATH = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
    # exp_config = ExperimentConfig(ROOT_PATH, 4, Z_DIM, [64, N_2, N_3],
    #                               ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW)
    # BATCH_SIZE = exp_config.BATCH_SIZE
    # DATASET_NAME = exp_config.dataset_name
    # exp_config.check_and_create_directories(run_id, create=False)
    #
    # iterator, val_images, val_labels, val_annotations = load_images(exp_config,
    #                                                                 dataset_type="val")
    # print("Images shape={}".format(val_images.shape))
    # print("Labels shape={}".format(val_labels.shape))
    # print("Manual Annotations shape={}".format(val_annotations.shape))

    # Test cases for load_images
    # train_val_iterator, images, labels, manual_annotation = load_images(exp_config, "train",
    #                                                                     exp_config.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS)


