import os
from glob import glob

import cv2
import numpy as np
from PIL import Image

from clearn.dao.idao import IDao


class DriveDao(IDao):
    def __init__(self,
                 split_name: str,
                 num_validation_samples: int):
        self.dataset_name = "drive_processed"
        self.split_name = split_name
        self.num_validation_samples = num_validation_samples

    @property
    def number_of_training_samples(self):
        return 20 - self.num_validation_samples

    @property
    def number_of_testing_samples(self):
        return 20

    @property
    def image_shape(self):
        return [512, 512, 3]

    @property
    def max_value(self):
        return 255.

    @property
    def num_classes(self):
        return 2

    def load_test_1(self, data_dir):
        test_x_path = f"{data_dir}/test/image/"
        test_y_path = f"{data_dir}/test/mask/"
        test_x_filenames, test_y_filenames = DriveDao._get_image_files(test_x_path, test_y_path)
        return DriveDao._load_images_label(test_x_filenames, test_y_filenames, test_x_path, test_y_path)

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

    @staticmethod
    def _load_images_label(x_path, y_path):
        x = np.zeros(( 20, 512, 512, 3))
        image_num = 0
        for file in glob(x_path):
            print(file)
            image = Image.open(file)
#            gt_file = file.replace("image", "mask")
#            print(gt_file)
#            gt = Image.open(gt_file)
            image = np.asarray(image)
            # gt = np.asarray(gt)

            print(image.shape)
            # print(gt.shape)
            x[image_num] = image
#            y[image_num ] = gt
            image_num += 1
        print(x.shape)
        y = np.zeros(x.shape[0])
        return x, y

    def load_train_images_and_label(self, data_dir,
                                    map_filename=None,
                                    training_phase=None):
        train_x_path = f"{data_dir}/train/image/*.jpg"
        train_y_path = f"{data_dir}/train/mask/*.jpg"
        return DriveDao._load_images_label(train_x_path, train_y_path)
