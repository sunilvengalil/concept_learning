import os
import gzip
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import json

from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao


def load_images(_config, dataset_type="train", manual_annotation_file=None):

    train_val_data_iterator = TrainValDataIterator.from_existing_split(_config.split_name,
                                                                       _config.DATASET_PATH,
                                                                       _config.BATCH_SIZE,
                                                                       manual_annotation_file=manual_annotation_file,
                                                                       dao=MnistDao
                                                                       )
    num_images = train_val_data_iterator.get_num_samples(dataset_type)
    feature_shape = list(train_val_data_iterator.get_feature_shape())
    num_images = (num_images // _config.BATCH_SIZE) * _config.BATCH_SIZE
    feature_shape.insert(0, num_images)
    train_images = np.zeros(feature_shape)
    i = 0
    # TODO remove hard coding
    train_labels = np.zeros([num_images, ])
    manual_annotations = np.zeros([num_images, 11])

    train_val_data_iterator.reset_counter(dataset_type)
    while train_val_data_iterator.has_next(dataset_type):
        batch_images, batch_labels, batch_annotations = train_val_data_iterator.get_next_batch(dataset_type)
        if batch_images.shape[0] < _config.BATCH_SIZE:
            break
        train_images[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_images
        train_labels[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_labels
        manual_annotations[i * _config.BATCH_SIZE:(i + 1) * _config.BATCH_SIZE, :] = batch_annotations
        i += 1
    return train_val_data_iterator, train_images, train_labels, manual_annotations


class TrainValDataIterator:
    VALIDATION_Y_RAW = "validation_y_raw"
    VALIDATION_Y_ONE_HOT = "validation_y"
    VALIDATION_X = "validation_x"
    TRAIN_Y = "train_y"
    TRAIN_X = "train_x"

    @classmethod
    def load_manual_annotation(manual_annotation_file):
        df = pd.read_csv(manual_annotation_file)
        return df.values

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
        train_y = np.asarray(data.reshape(data.shape[0])).astype(np.int)

        val = dataset_dict["validation"]
        val_x = val[x_columns].values
        val_x = val_x.reshape(train_x.shape)

        val_y = val[['label']].values
        val_y = np.asarray(val_y.reshape(val_y.shape[0])).astype(np.int)

        if len(split_names) != 2:
            raise Exception("Split not implemented for for than two splits")

        #TODO change this to numpy - remove the for loop performance improvement
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
                            split_name,
                            split_location,
                            batch_size=None,
                            manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                            manual_annotation_file=None,
                            init_config=None,
                            dao:IDao=MnistDao()):
        """
        Creates and initialize an instance of TrainValDataIterator
        @param: split_name:Name of the train/valid/test split
        @param: split_location: path to folder where dataset split(train/val/test) is stored
        @param: batch_size: number of samples in each batch
        @param: manual_labels_config:
        """
        instance = cls(batch_size=batch_size, dao=dao)
        # TODO convert this to lazy loading/use generator
        instance.dataset_dict = dao.load_train_val_existing_split(split_name, split_location)

        if init_config is not None and "val_y" in init_config:
            instance.val_y = instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
        instance.dataset_dict = instance.dataset_dict
        # instance.dataset_dict = cls.load_train_val_existing_split(split_name, split_location)

        instance.train_x = instance.dataset_dict[TrainValDataIterator.TRAIN_X]
        instance.train_y = instance.dataset_dict[TrainValDataIterator.TRAIN_Y]
        instance.val_x = instance.dataset_dict[TrainValDataIterator.VALIDATION_X]
        instance.val_y = instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
        # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
        instance.unique_labels = np.unique(instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_RAW])
        instance.manual_labels_config = manual_labels_config
        _manual_annotation = None
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
                _manual_annotation = np.random.choice(instance.unique_labels, len(instance.train_x))
        instance.get_manual_annotation(manual_annotation_file, _manual_annotation=_manual_annotation)

        instance.train_idx = 0
        instance.val_idx = 0
        return instance

    def get_manual_annotation(self, manual_annotation_file, _manual_annotation):
        if self.manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
            self.manual_annotation = np.zeros((len(_manual_annotation), 11), dtype=np.float)
            if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                for i, label in enumerate(_manual_annotation):
                    self.manual_annotation[i, int(_manual_annotation[i, 0])] = 1.0
                    self.manual_annotation[i, 10] = _manual_annotation[i, 1]
            else:
                for i, label in enumerate(_manual_annotation):
                    self.manual_annotation[i, _manual_annotation[i]] = 1.0
                    self.manual_annotation[i, 10] = 0 # set manual annotation confidence as 0
        elif self.manual_labels_config == ExperimentConfig.USE_ACTUAL:
            self.manual_annotation = np.zeros((len(self.train_x), 11), dtype=np.float)

            self.manual_annotation[:, 0:10] = self.train_y
            self.manual_annotation[:, 10] = 1 # set manual annotation confidence as 1

    def __init__(self, dataset_path=None,
                 shuffle=False,
                 stratified=None,
                 validation_samples=128,
                 split_location=None,
                 split_names=[],
                 batch_size=None,
                 manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                 manual_annotation_file=None,
                 dao:IDao = MnistDao()):
        self.train_idx = 0
        self.val_idx = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dao = dao
        if dataset_path is not None :
            self.entire_data_x, self.entire_data_y = load_train(dataset_path, dao=dao)
            if validation_samples == -1:
                percentage_to_be_sampled = 0.3
            else:
                percentage_to_be_sampled = validation_samples / len(self.entire_data_y)

            self.dataset_dict = load_train_val(dataset_path, shuffle=shuffle,
                                               stratified=stratified,
                                               percentage_to_be_sampled=percentage_to_be_sampled,
                                               split_location=split_location,
                                               split_names=split_names,
                                               dao=dao)
            # TODO remove this later start using dataset_dict instead
            self.train_x = self.dataset_dict[TrainValDataIterator.TRAIN_X]
            self.train_y = self.dataset_dict[TrainValDataIterator.TRAIN_Y]
            self.val_x = self.dataset_dict[TrainValDataIterator.VALIDATION_X]
            self.val_y = self.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
            # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
            self.unique_labels = np.unique(self.dataset_dict["validation_y_raw"])
            self.manual_labels_config = manual_labels_config
            _manual_annotation = None
            if manual_labels_config == ExperimentConfig.USE_CLUSTER_CENTER:
                if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                    _manual_annotation = TrainValDataIterator.load_manual_annotation()
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
                    _manual_annotation = np.random.choice(self.unique_labels, len(self.train_x))

            self.get_manual_annotation(manual_annotation_file, _manual_annotation=_manual_annotation)

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
            self.train_idx += 1
        elif dataset_type == "val":
            x = self.val_x[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            y = self.val_y[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            label = self.manual_annotation[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
            # TODO check if this is last batch, if yes,reset the counter
            self.val_idx += 1
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")
        return x, y, label

    def get_num_samples(self, dataset_type):
        if dataset_type == "train":
            return len(self.train_x)
        elif dataset_type == "val":
            return len(self.val_x)
        else:
            raise ValueError("dataset_type should be either 'train' or 'val' ")

    def reset_counter(self, dataset_type):
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


def load_train(data_dir,
               shuffle=True,
               dao:IDao=MnistDao()):
    return dao.load_train(data_dir, shuffle)


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


def load_train_val(data_dir, shuffle=False,
                   stratified=None,
                   percentage_to_be_sampled=0.7,
                   split_location=None,
                   split_names=[],
                   dao: IDao=MnistDao()):
    return dao.load_train_val(data_dir,
                              shuffle,
                              stratified,
                              percentage_to_be_sampled,
                              split_location, split_names)


if __name__ == "__main__":
    # Test cases for load_images
    from clearn.config import ExperimentConfig
    N_3 = 16
    N_2 = 128
    Z_DIM = 20
    run_id = 1

    ROOT_PATH = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
    exp_config = ExperimentConfig(ROOT_PATH, 4, Z_DIM, [64, N_2, N_3],
                                  ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW)
    BATCH_SIZE = exp_config.BATCH_SIZE
    DATASET_NAME = exp_config.dataset_name
    exp_config.check_and_create_directories(run_id, create=False)

    iterator, val_images, val_labels, val_annotations = load_images(exp_config,
                                                                    dataset_type="val")
    print("Images shape={}".format(val_images.shape))
    print("Labels shape={}".format(val_labels.shape))
    print("Manual Annotations shape={}".format(val_annotations.shape))

    # Test cases for load_images
    # train_val_iterator, images, labels, manual_annotation = load_images(exp_config, "train",
    #                                                                     exp_config.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS)
