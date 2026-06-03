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
z_dim = 16
run_id = 210526
num_units=[64, 128,32,2]
strides = [2, 2, 1,2]
num_dense_layers = 0
create_split = False
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_ELBOW
experiment_name = "premi"
num_clusters = 10
num_epochs = 10
dataset_name = "mnist"
beta = 0
use_global_average_pooling = True
fully_convolutional = False

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
                              log_level = logging.DEBUG,
                              use_global_average_pooling=use_global_average_pooling,
                              fully_convolutional= False
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

tf.compat.v1.reset_default_graph()
train_val_data_iterator, exp_config, model = initialize_model_train_and_get_features(experiment_name=experiment_name,
                                                                                     root_path=root_path,
                                                                                     z_dim=z_dim,
                                                                                     strides=strides,
                                                                                     num_dense_layers=num_dense_layers,
                                                                                     run_id=run_id,
                                                                                     create_split=create_split,
                                                                                     num_epochs=num_epochs,
                                                                                     num_epochs_completed=0,
                                                                                     num_cluster_config=num_cluster_config,
                                                                                     model_type=model_type,
                                                                                     num_decoder_layer=4,
                                                                                     num_units=num_units,
                                                                                     confidence_decay_factor=confidence_decay_factor,
                                                                                     beta=beta,
                                                                                     supervise_weight=150,
                                                                                     dataset_name=exp_config.dataset_name,
                                                                                     split_name=exp_config.split_name,
                                                                                     batch_size=64,
                                                                                     eval_interval_in_epochs=1,
                                                                                     num_val_samples=exp_config.num_val_samples,
                                                                                     manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                                                                     reconstruction_weight=1,
                                                                                     learning_rate = 1e-3,
                                                                                     log_level = logging.DEBUG,
                                                                                     dao = dao,
                                                                                     use_global_average_pooling=use_global_average_pooling,
                                                                                     fully_convolutional=fully_convolutional
                                                                                    )
tf.compat.v1.reset_default_graph()