# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf

from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.models.classify.classifier import ClassifierModel
from clearn.utils.data_loader import DataIterator
from clearn.models.architectures.custom.tensorflow_graphs import cnn_n_layer


class SupervisedClassifierModel(ClassifierModel):
    _model_name_ = "SupervisedClassifierModel"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess: tf.Session,
                 epoch: int,
                 dao: IDao,
                 num_units_in_layer=None,
                 check_point_epochs=None,
                 test_data_iterator: DataIterator = None,
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         check_point_epochs=check_point_epochs,
                         dao=dao,
                         test_data_iterator=test_data_iterator)
        self.label_dim = dao.num_classes
        self.strides = [2] * len(num_units_in_layer)
        if num_units_in_layer is None or len(num_units_in_layer) == 0:
            self.n = [128, 64, 32, exp_config.Z_DIM]
        else:
            self.n = num_units_in_layer
        self.sample_num = 64  # number of generated images to be saved
        self.num_images_per_row = 4  # should be a factor of sample_num
        self.images = None
        # graph inputs for visualize training results
        if epoch != -1:
            self.epoch = epoch

    def _encoder(self, x, reuse=False):
        return cnn_n_layer(self, x, self.exp_config.Z_DIM, reuse)
