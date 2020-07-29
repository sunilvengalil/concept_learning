import os
import gzip
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import json


def load_images(_config, dataset_type="train", manual_annotation_file=None):
    train_val_data_iterator = TrainValDataIterator.from_existing_split(_config.split_name,
                                                                       _config.DATASET_PATH,
                                                                       _config.BATCH_SIZE,
                                                                       manual_annotation_file=manual_annotation_file
                                                                       )
    num_images = train_val_data_iterator.get_num_samples(dataset_type)
    feature_shape = list(train_val_data_iterator.get_feature_shape())
    num_images = (num_images // _config.BATCH_SIZE) * _config.BATCH_SIZE
    feature_shape.insert(0, num_images)
    train_images = np.zeros(feature_shape)
    i = 0
    # TODO remove hard coding
    train_labels = np.zeros([num_images, 10])
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
    USE_ACTUAL = "USE_ACTUAL"
    USE_CLUSTER_CENTER = "USE_CLUSTER_CENTER"
    VALIDATION_Y_RAW = "validation_y_raw"
    VALIDATION_Y_ONE_HOT = "validation_y"
    VALIDATION_X = "validation_x"
    TRAIN_Y = "train_y"
    TRAIN_X = "train_x"

    @classmethod
    def _load_manual_annotation(cls, manual_annotation_file):
        df = pd.read_csv(manual_annotation_file)
        return df.values

    @classmethod
    def load_train_val_existing_split(cls, split_name, split_location):
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
        train_x = data.reshape((data.shape[0], 28, 28, 1))
        data = train[['label']].values
        train_y = np.asarray(data.reshape(data.shape[0])).astype(np.int)

        val = dataset_dict["validation"]
        val_x = val[x_columns].values
        val_x = val_x.reshape((val_x.shape[0], 28, 28, 1))

        val_y = val[['label']].values
        val_y = np.asarray(val_y.reshape(val_y.shape[0])).astype(np.int)

        if len(split_names) != 2:
            raise Exception("Split not implemented for for than two splits")

        _val_y = np.zeros((len(val_y), 10), dtype=np.float)
        for i, label in enumerate(val_y):
            _val_y[i, val_y[i]] = 1.0

        _train_y = np.zeros((len(train_y), 10), dtype=np.float)
        for i, label in enumerate(train_y):
            _train_y[i, train_y[i]] = 1.0

        # TODO separate normalizing and loading logic
        return {TrainValDataIterator.TRAIN_X: train_x / 255.,
                TrainValDataIterator.TRAIN_Y: _train_y,
                TrainValDataIterator.VALIDATION_X: val_x / 255.,
                TrainValDataIterator.VALIDATION_Y_ONE_HOT: _val_y,
                TrainValDataIterator.VALIDATION_Y_RAW: val_y}

    """
    Creates and initialize an instance of TrainValDataIterator
    @param: init_config:list A list of attributes that needs to be initialized
    """
    @classmethod
    def from_existing_split(cls,
                            split_name,
                            split_location,
                            batch_size=None,
                            manual_labels_config=USE_CLUSTER_CENTER,
                            manual_annotation_file=None,
                            init_config=None):
        instance = cls()
        instance.batch_size = batch_size
        # TODO convert this to lazy loading
        dataset_dict = cls.load_train_val_existing_split(split_name, split_location)

        if init_config is not None and "val_y" in init_config:
            instance.val_y = dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
        instance.dataset_dict = dataset_dict
        instance.dataset_dict = cls.load_train_val_existing_split(split_name, split_location)

        instance.train_x = instance.dataset_dict[TrainValDataIterator.TRAIN_X]
        instance.train_y = instance.dataset_dict[TrainValDataIterator.TRAIN_Y]
        instance.val_x = instance.dataset_dict[TrainValDataIterator.VALIDATION_X]
        instance.val_y = instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
        instance.unique_labels = np.unique(instance.dataset_dict[TrainValDataIterator.VALIDATION_Y_RAW])
        instance.manual_labels_config = manual_labels_config
        if manual_labels_config == TrainValDataIterator.USE_CLUSTER_CENTER:
            if manual_annotation_file is not None and os.path.isfile(manual_annotation_file):
                _manual_annotation = cls._load_manual_annotation(manual_annotation_file)
                print("Loaded manual annotation")
                print(f"Number of samples with manual confidence {sum(_manual_annotation[:, 1] > 0)}")
                instance.manual_annotation = np.zeros((len(_manual_annotation), 11), dtype=np.float)
                for i, label in enumerate(_manual_annotation):
                    instance.manual_annotation[i, int(_manual_annotation[i, 0])] = 1.0
                    instance.manual_annotation[i, 10] = _manual_annotation[i, 1]
            else:
                # TODO if we are using random prior with uniform distribution, do we need to keep
                # manual confidence as 0.5 or 0
                print("Warning", "{} path does not exist. Creating random prior with uniform distribution".
                      format(manual_annotation_file))
                _manual_annotation = np.random.choice(instance.unique_labels, len(instance.train_x))
                instance.manual_annotation = np.zeros((len(_manual_annotation), 11), dtype=np.float)
                for i, label in enumerate(_manual_annotation):
                    instance.manual_annotation[i, _manual_annotation[i]] = 1.0
                    instance.manual_annotation[i, 10] = 0
        elif manual_labels_config == TrainValDataIterator.USE_ACTUAL:
            instance.manual_annotation = np.zeros((len(instance.train_x), 11), dtype=np.float)

            instance.manual_annotation[:, 0:10] = instance.train_y
            instance.manual_annotation[:, 10] = 1

        instance.train_idx = 0
        instance.val_idx = 0
        # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
        return instance

    def __init__(self, dataset_path=None, shuffle=False,
                 stratified=None,
                 validation_samples=100,
                 split_location=None,
                 split_names=[],
                 batch_size=None,
                 manual_annotation_path=None):
        self.train_idx = 0
        self.val_idx = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        if dataset_path is not None :
            self.entire_data_x, self.entire_data_y = load_train(dataset_path)
            percentage_to_be_sampled = validation_samples / len(self.entire_data_y)
            self.dataset_dict = load_train_val(dataset_path, shuffle=shuffle,
                                               stratified=stratified,
                                               percentage_to_be_sampled=percentage_to_be_sampled,
                                               split_location=split_location,
                                               split_names=split_names)
            # TODO remove this later start using dataset_dict instead
            self.train_x = self.dataset_dict[TrainValDataIterator.TRAIN_X]
            self.train_y = self.dataset_dict[TrainValDataIterator.TRAIN_Y]
            self.val_x = self.dataset_dict[TrainValDataIterator.VALIDATION_X]
            self.val_y = self.dataset_dict[TrainValDataIterator.VALIDATION_Y_ONE_HOT]
            # TODO fix this later set unique labels based on the shape of the smallest dataset in the dict
            self.unique_labels = np.unique(self.dataset_dict["validation_y_raw"])

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


def load_train(data_dir, directory_for_sample_images = None,
               shuffle=True):
    data_dir = os.path.join(data_dir, "images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.float)

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    tr_y = data.reshape((60000, 28, 28, 1))

    #code to save images
    if directory_for_sample_images is not None:
        for i in range(10):
            cv2.imwrite(directory_for_sample_images+ str(i) + '.jpg', tr_y[i])

    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    tr_y = data.reshape((60000))
    tr_y = np.asarray(tr_y).astype(np.int)
    if shuffle:
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(tr_y)
        np.random.seed(seed)
        np.random.shuffle(tr_y)

    y_vec = np.zeros((len(tr_y), 10), dtype=np.float)
    for i, label in enumerate(tr_y):
        y_vec[i, tr_y[i]] = 1.0

    return tr_y / 255., y_vec


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
                   split_names=[]):

    data_dir = os.path.join(data_dir, "images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            _data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return _data

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    x = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    y = np.asarray(data.reshape(60000)).astype(np.int)

    seed = 547
    _stratify = None
    if stratified:
        _stratify = y

    if len(split_names) == 2:
        splitted = train_test_split(x, y, test_size=percentage_to_be_sampled,
                                    stratify=_stratify, shuffle=shuffle,
                                    random_state=seed)
        train_x = splitted[0]
        val_x = splitted[1]
        train_y = splitted[2]
        val_y = splitted[3]

        # TODO change this to save only indices.
        # Alternately save the seed after verifying that same seed generates same split
        if split_location[-1] == "/":
            split_name = split_location[:-1].rsplit("/", 1)[1]
        else:
            split_name = split_location.rsplit("/", 1)[1]
        dataset_dict = {}
        num_splits = len(split_names)
        dataset_dict["split_names"] = split_names

        for split_num, split in enumerate(split_names):
            # TODO remove hard coding of dimensions below
            train_df = pd.DataFrame(splitted[split_num].reshape(splitted[split_num].shape[0], 28 * 28))
            train_df["label"] = splitted[split_num + num_splits]
            train_df.to_csv(split_location + split + ".csv", index=False)
        print(split_location)
        json_ = split_location + split_name + ".json"
        with open(json_, "w") as fp:
            print("Writing json to ", json_)
            json.dump(dataset_dict, fp)
        print("Writing json to ", json_)
    else:
        raise Exception("Split not implemented for for than two splits")

    _val_y = np.zeros((len(val_y), 10), dtype=np.float)
    for i, label in enumerate(val_y):
        _val_y[i, val_y[i]] = 1.0

    _train_y = np.zeros((len(train_y), 10), dtype=np.float)
    for i, label in enumerate(train_y):
        _train_y[i, train_y[i]] = 1.0
    # TODO separate normalizing and loading logic
    return {TrainValDataIterator.TRAIN_X: train_x / 255.,
            TrainValDataIterator.TRAIN_Y: _train_y,
            TrainValDataIterator.VALIDATION_X: val_x / 255.,
            TrainValDataIterator.VALIDATION_Y_ONE_HOT: _val_y,
            TrainValDataIterator.VALIDATION_Y_RAW: val_y}


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
