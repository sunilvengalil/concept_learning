import math
from math import log
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

import matplotlib
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import tensorflow as tf

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

import logging

from tests.test_model_apis import root_path

cluster_column_name ="cluster_level_1"
cluster_column_name_2 ="cluster_level_2"
cluster_column_name_3 ="cluster_level_3"

z_dim = 10
run_id = 1
num_units=[16, 32]
strides = [2, 2, 1]
num_dense_layers = 1
create_split = True
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_ELBOW
experiment_name = "premi"
base_path = f"/Users/sunil/concept_learning_exp/{experiment_name}/Exp_0_{num_units[0]}_{z_dim}_ELBOW_{run_id}"
print(base_path)


num_epochs = 10
manual_annotation_file = f"manual_annotation_epoch_{num_epochs - 1:.1f}.csv"
dataset_name = "mnist"
split_name = "Split_1"
num_val_samples = 128
model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST

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
                              confidence_decay_factor=run_id,
                              beta=5,
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


K = exp_config.confidence_decay_factor
def convert_distance_to_confidence_exp(dist):
    return np.exp(-1 / K * dist)

def get_distance_exp(confidence):
    return -K * log(confidence)

def convert_distance_to_confidence(dist):
    return np.exp(-1 / K * dist * dist)

def get_percentage_correct(confidence:float, df:pd.DataFrame):
    df1 = df[df["manual_annotation_confidence"] > confidence]
    if df1.shape[0] != 0:
        df2 = df1[df1["manual_annotation"] == df1["label"]]
        return df2.shape[0] / df1.shape[0]
    else:
        return 1

def get_distance(confidence):
    return math.sqrt(-K * log(confidence))



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
                                                                                     confidence_decay_factor=run_id,
                                                                                     beta=5,
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
                                                                                     dao = dao
                                                                                    )
tf.compat.v1.reset_default_graph()