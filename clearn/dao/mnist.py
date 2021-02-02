import numpy as np
import os
import gzip
from clearn.dao.idao import IDao


class MnistDao(IDao):

    @property
    def number_of_training_samples(self):
        return 60000

    @property
    def number_of_testing_samples(self):
        return 10000

    @property
    def image_shape(self):
        return [28, 28, 1]

    def __init__(self, split_name="Split_1"):
        self.dataset_name = "mnist"
        self.split_name = split_name

    @property
    def max_value(self):
        return 255.

    @property
    def num_classes(self):
        return 10

    def load_test_1(self, data_dir):
        data_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(data_dir + 't10k-images-idx3-ubyte.gz',
                                 self.number_of_testing_samples,
                                 16,
                                 28 * 28)
        x = data.reshape((self.number_of_testing_samples, 28, 28, 1))
        data = self.extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        y = np.asarray(data.reshape(self.number_of_testing_samples)).astype(np.int)
        return x, y

    def load_train(self, data_dir, shuffle):
        tr_x, tr_y = self.load_train_val_1(data_dir)
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

    def load_train_val_1(self, data_dir):
        data_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(data_dir + 'train-images-idx3-ubyte.gz',
                                 self.number_of_training_samples,
                                 16,
                                 28 * 28)
        x = data.reshape((self.number_of_training_samples, 28, 28, 1))

        data = self.extract_data(data_dir + '/train-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        y = np.asarray(data.reshape(self.number_of_training_samples)).astype(np.int)
        return x, y
