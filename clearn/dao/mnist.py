import numpy as np
import os
import gzip
from clearn.dao.idao import IDao


class MnistDao(IDao):
    def __init__(self,
                 split_name: str,
                 num_validation_samples: int,add_invalid_images=False):
        self.dataset_name = "mnist"
        self.split_name = split_name
        self.num_validation_samples = num_validation_samples
        self.add_invalid_images=add_invalid_images
        super().__init__()

    @property
    def number_of_training_samples(self):
        if self.data_dict is not None and self.TRAIN_X in self.data_dict.keys() and self.data_dict[self.TRAIN_X] is not None and len(self.data_dict[self.TRAIN_X]) > 0:
            return len(self.data_dict[self.TRAIN_X])
        return 50000 - self.num_validation_samples

    @property
    def number_of_testing_samples(self):
        return 10000

    @property
    def image_shape(self):
        return [28, 28, 1]

    @property
    def max_value(self):
        return 255.

    @property
    def num_classes(self):
        if self.add_invalid_images:
            return 11
        else:
            return 10

    def load_test_1(self, data_dir):
        images_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(images_dir + 't10k-images-idx3-ubyte.gz',
                                 self.number_of_testing_samples,
                                 16,
                                 28 * 28)
        x = data.reshape((self.number_of_testing_samples, 28, 28, 1))
        data = self.extract_data(images_dir + '/t10k-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        y = np.asarray(data.reshape(self.number_of_testing_samples)).astype(int)
        if self.add_invalid_images:
            invalid_images = self.load_invalid_images(os.path.join(data_dir, "invalid_images.npy"))[0:self.number_of_testing_samples//10 + 1];
            x = np.concatenate((x, invalid_images), axis=0)
            y = np.concatenate((y,np.ones(invalid_images.shape[0],np.int16) * 10), axis=0)

        return x, y

    def load_train(self, data_dir, shuffle, split_location=None):
        tr_x, tr_y = self.load_train_images_and_label(data_dir)
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
            _data = np.frombuffer(buf, dtype=np.uint8).astype(float)
        return _data

    def load_train_images_and_label(self, data_dir, map_filename=None, training_phase=None):
        images_dir = os.path.join(data_dir, "images/")
        data = self.extract_data(images_dir + 'train-images-idx3-ubyte.gz',
                                 self.number_of_training_samples,
                                 16,
                                 28 * 28)
        print("Mnist x shape after loading", data.shape)

        x = data.reshape((-1, 28, 28, 1))
        print("Mnist x shape after reshaping", x.shape)

        data = self.extract_data(images_dir + '/train-labels-idx1-ubyte.gz', self.number_of_training_samples, 8, 1)
        print("Mnist y shape after loading", data.shape)

        y = np.asarray(data.reshape(-1)).astype(int)
        print("Mnist y shape after reshaping", y.shape)
        if self.add_invalid_images:
            invalid_images = self.load_invalid_images(os.path.join(data_dir, "invalid_images.npy"));
            x = np.concatenate((x, invalid_images), axis=0)
            y = np.concatenate((y,np.ones(invalid_images.shape[0]) * 10), axis=0)
        return x, y

    @staticmethod
    def load_invalid_images(filename):
        loaded_array = np.load(filename)
        print(f"Loaded shape:   {loaded_array.shape}")
        return loaded_array
