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
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
import tensorflow as tf

from clearn.utils.data_loader import load_images
from clearn.utils.utils import get_latent_vector_column, show_all_variables, get_pmf_y_given_z
from clearn.config.common_path import get_encoded_csv_file
from clearn.config import ExperimentConfig
from clearn.experiments.experiment import Experiment
from clearn.analysis.encode_decode import decode
from clearn.analysis.cluster_utils import cluster_and_decode_latent_vectors, display_images,get_cluster
from clearn.analysis.cluster_utils import get_cluster_groups,assign_manual_label_and_confidence, get_samples_for_cluster
from clearn.analysis import ManualAnnotation, Cluster, ClusterGroup
from PIL import Image
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import load_trained_model, MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,get_train_val_iterator


experiment_name = "Experiment_4"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim = 32
learning_rate = 0.001
num_epochs = 5
num_runs = 1
create_split = True
completed_z_dims = 0
completed_runs = 0
run_id = 3
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
num_units = [64, 128, 64, 64]
# num_units = [128, 256, 512, 1024]
train_val_data_iterator = None
beta = 0
supervise_weight = 0
dataset_name = "cifar_10"
split_name = "split_1"
dao = get_dao(dataset_name, split_name, 128)


tf.reset_default_graph()
model, exp_config, _, _, epochs_completed = load_trained_model(experiment_name=experiment_name,
                                                               z_dim=z_dim,
                                                               run_id=run_id,
                                                               num_cluster_config=num_cluster_config,
                                                               manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                                               supervise_weight=0,
                                                               beta=0,
                                                               reconstruction_weight=1,
                                                               model_type=MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,
                                                               num_units=num_units,
                                                               save_reconstructed_images=True,
                                                               split_name="split_1",
                                                               num_val_samples=128,
                                                               learning_rate=0.001,
                                                               dataset_name="cifar_10",
                                                               activation_output_layer="LINEAR",
                                                               write_predictions=True,
                                                               seed=547,
                                                               root_path=root_path,
                                                               eval_interval=300,
                                                               run_evaluation_during_training=False
                                                               )
print("Number of epochs completed", model.num_training_epochs_completed)

exp_config.check_and_create_directories(run_id)
cluster_column_name ="cluster_level_1"
cluster_column_name_2 ="cluster_level_2"

from clearn.analysis.encode_images import encode_images

manual_annotation_file = os.path.join(exp_config.ANALYSIS_PATH,
                                      "manual_annotation_epoch_{}.csv".format(epochs_completed - 1)
                                     )
train_val_iterator = get_train_val_iterator(create_split,
                                            dao, exp_config,
                                            model.num_training_epochs_completed,
                                            exp_config.split_name)
images, labels, manual_annotation_np = load_images(exp_config,train_val_iterator,
                                                                    "train",
                                                                    manual_annotation_file)
unique_labels = train_val_iterator.get_unique_labels()
num_batches = images.shape[0] / exp_config.BATCH_SIZE
print("Number of epochs completed {}".format(model.num_training_epochs_completed))

encoded_df = encode_images(model=model, train_val_data_iterator=train_val_iterator, dataset_type="train")
filename  = get_encoded_csv_file(exp_config, model.num_training_epochs_completed, "train")
mean_col_names, sigma_col_names, z_col_names,_ = get_latent_vector_column(exp_config.Z_DIM)
df = pd.read_csv(os.path.join(exp_config.ANALYSIS_PATH, filename))
z_min = df[z_col_names].min().min()
z_max = df[z_col_names].max().max()
latent_vectors = df[z_col_names].values
print("run_id={} z_min={} z_max={}".format(run_id, z_min, z_max))
print("Latent vectors shape",latent_vectors.shape)
