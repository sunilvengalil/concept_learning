import math
from math import log
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

import json
import cv2

import matplotlib
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
import tensorflow as tf

from tests.test_model_apis import root_path

from clearn.utils.data_loader import load_images, TrainValDataIterator
from clearn.utils.utils import get_latent_vector_column, show_all_variables, get_pmf_y_given_z
from clearn.config.common_path import get_encoded_csv_file
from clearn.config import ExperimentConfig
from clearn.experiments.experiment import Experiment, initialize_model_train_and_get_features, get_model, get_train_val_iterator,  MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST
from clearn.analysis.encode_decode import decode
from clearn.analysis.cluster_utils import cluster_and_decode_latent_vectors,get_cluster
from clearn.analysis.cluster_utils import get_cluster_groups, get_samples_for_cluster
from clearn.analysis import ManualAnnotation, Cluster, ClusterGroup
from clearn.utils.data_loader import TrainValDataIterator
from clearn.dao.dao_factory import get_dao
from clearn.analysis.cluster_utils import cluster_next_level, plot_number_of_samples_vs_label, compute_distance, cluster_next_level_gmm, cluster_and_decode_latent_vectors_gmm
from clearn.analysis.cluster_utils import assign_manual_label_and_confidence, plot_distance_distribution, compute_distance_level_2, display_images
from PIL import Image
from typing import List
import logging

import logging

env = "colab"
z_dim = 10
run_id = 20526
num_units=[64, 128,32]
strides = [2, 2, 1]
num_dense_layers = 1
create_split = False
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_ELBOW
experiment_name = "premi"
num_clusters = 10
num_epochs = 10
dataset_name = "mnist"

split_name = "Split_1"
num_val_samples = 128
model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST
confidence_decay_factor=0.8

dao = get_dao(dataset_name,
              split_name,
              num_val_samples,
              dataset_path=os.path.join(root_path, "datasets/"),
                               concept_id=-1)

exp_config = ExperimentConfig(root_path=root_path,
                              num_decoder_layer=4,
                              strides=strides,
                              num_dense_layers=num_dense_layers,
                              z_dim=z_dim,
                              num_units=num_units,
                              num_cluster_config=num_cluster_config,
                              confidence_decay_factor=confidence_decay_factor,
                              beta=0,
                              dao=dao,
                              supervise_weight=150,
                              dataset_name=dataset_name,
                              split_name=split_name,
                              model_name="VAE",
                              batch_size=64,
                              eval_interval_in_epochs=1,
                              name=experiment_name,
                              num_val_samples=num_val_samples,
                              manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                              reconstruction_weight = 1,
                              activation_hidden_layer = "RELU",
                              activation_output_layer = "SIGMOID",
                              learning_rate = 1e-3,
                              log_level = logging.DEBUG
                              )
exp_config.check_and_create_directories(run_id)
BATCH_SIZE = exp_config.BATCH_SIZE
DATASET_NAME = exp_config.dataset_name


from clearn.analysis.encode_decode import decode, encode, encode_and_get_features, decode_and_get_features

train_val_data_iterator = get_train_val_iterator(create_split=create_split,
                                                 dao= dao,
                                                 exp_config= exp_config,
                                                 num_epochs_completed=0,
                                                 split_name=exp_config.split_name)
images, labels, _ = load_images(exp_config,
                                train_val_data_iterator,
                                "train"
                               )

model = None
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
    model = get_model(dao = dao,
              exp_config=exp_config,
              model_type=model_type,
              num_epochs=10,
              sess=sess,
              test_data_iterator=None,
              train_val_data_iterator=None)

    hidden_feature_names, mus, sigmas, latent_vectors, features = encode_and_get_features(model,
                                                                                        images[0:1],
                                                                                        exp_config.BATCH_SIZE,
                                                                                        exp_config.Z_DIM, [3])
tf.compat.v1.reset_default_graph()


print(hidden_feature_names)
print(features[0][4][0].shape)


