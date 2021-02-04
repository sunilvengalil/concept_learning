from typing import *
import numpy as np
from clearn.dao.idao import IDao
from tensorflow.keras import datasets
import ssl
class CiFar10Dao(IDao):
    def __init__(self, split_name: str):
        self.dataset_name = "cifar_10"
        self.split_name = split_name

    @property
    def number_of_training_samples(self) -> int:
        return 50000

    @property
    def image_shape(self) -> Tuple[int]:
        return[32, 32, 3]

    @property
    def max_value(self) -> int:
        return 255.

    @property
    def num_classes(self) -> int:
        return 10

    def load_train(self, data_dir, shuffle):
        tr_x, tr_y = self.load_train_val_1(data_dir)

        if shuffle:
            seed = 547
            np.random.seed(seed)
            np.random.shuffle(tr_y)
            np.random.seed(seed)
            np.random.shuffle(tr_y)

        y_vec = np.eye(self.num_classes)[np.squeeze(tr_y)]

        return tr_x / self.max_value, y_vec

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def load_train_val_1(self, data_dir):
        ssl._create_default_https_context = ssl._create_unverified_context

        (tr_x, tr_y), (test_images, test_labels) = datasets.cifar10.load_data()
        # data = None
        # label = None
        # for batch_no in range(1, 6):
        #     batch_name = "data_batch_" + str(batch_no)
        #     data_dict = CiFar10Dao.unpickle(data_dir + "/cifar-10-batches-py" + "/" + batch_name)
        #     if data is None:
        #         data = data_dict[b"data"]
        #         label = data_dict[b"labels"]
        #     else:
        #         data = np.concatenate((data, data_dict[b"data"]), axis=0)
        #         label = np.concatenate((label, data_dict[b"labels"]), axis=0)
        #
        # tr_x, tr_y = self.reshape_x_and_y(data, label)
        return tr_x, np.squeeze(tr_y)

    def reshape_x_and_y(self, data, label):
        print(data.shape)
        x = data.reshape(
            (data.shape[0], self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        y = np.asarray(label).astype(np.int)
        print(y.shape)
        return x, y

    def load_test_1(self, data_dir):
        ssl._create_default_https_context = ssl._create_unverified_context
        (tr_x, tr_y), (test_images, test_labels) = datasets.cifar10.load_data()

        # batch_name = "test_batch"
        # data_dict = CiFar10Dao.unpickle(data_dir + "/cifar-10-batches-py" + "/" + batch_name)
        # data, label = data_dict[b"data"], data_dict[b"labels"]
        #return self.reshape_x_and_y(data, label)
        return test_images, np.squeeze(test_labels)
