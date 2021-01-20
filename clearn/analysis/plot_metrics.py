from typing import List
import os
import pandas as pd
import matplotlib.pyplot as plt
from clearn.config import ExperimentConfig
from sklearn.metrics import accuracy_score

from clearn.config.common_path import get_encoded_csv_file


def plot_z_dim_vs_accuracy(root_path: str,
                           experiment_name: str,
                           z_dim_range: List,
                           num_units: List,
                           num_cluster_config: str,
                           num_epochs: int,
                           run_id: int,
                           split_name: str,
                           num_val_samples: int,
                           manual_labels_config: str,
                           reconstruction_weight: float
                           ):
    exp_configs = dict()
    training_accuracies = []
    validation_accuracies = []
    for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
        exp_configs[z_dim] = ExperimentConfig(root_path=root_path,
                                              num_decoder_layer=4,
                                              z_dim=z_dim,
                                              num_units=num_units,
                                              num_cluster_config=num_cluster_config,
                                              confidence_decay_factor=5,
                                              beta=5,
                                              supervise_weight=150,
                                              dataset_name="mnist",
                                              split_name=split_name,
                                              model_name="VAE",
                                              batch_size=64,
                                              eval_interval=300,
                                              name=experiment_name,
                                              num_val_samples=num_val_samples,
                                              total_training_samples=60000,
                                              manual_labels_config=manual_labels_config,
                                              reconstruction_weight=reconstruction_weight,
                                              activation_hidden_layer="RELU",
                                              activation_output_layer="SIGMOID"
                                              )
        exp_config = exp_configs[z_dim]
        exp_config.check_and_create_directories(run_id)

        # Load training and validation results
        filename = get_encoded_csv_file(exp_config, int(num_epochs), "train")
        train_df = pd.read_csv(os.path.join(exp_config.ANALYSIS_PATH, filename))
        training_accuracies.append(accuracy_score(train_df["label_predicted"].values, train_df["label"].values))
        filename = get_encoded_csv_file(exp_config, int(num_epochs), "val")
        val_df = pd.read_csv(os.path.join(exp_config.ANALYSIS_PATH, filename))
        validation_accuracies.append(accuracy_score(val_df["label_predicted"].values, val_df["label"].values))

    z_dims = list(exp_configs.keys())
    plt.plot(z_dims, training_accuracies)
    plt.plot(z_dims, validation_accuracies)
    plt.legend(["train", "Validation"])
    plt.xlabel("z_dim")
    plt.ylabel("Accuracy")
    plt.title(f"Num units {num_units} Training epochs {num_epochs}")


def plot_epoch_vs_accuracy(root_path: str,
                           experiment_name: str,
                           epoch_range: List,
                           num_units: List,
                           num_cluster_config: str,
                           z_dim: int,
                           run_id: int,
                           data_set:List[str]=["train","val"]):
    exp_configs = dict()
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(epoch_range[0], epoch_range[1], epoch_range[2]):
        exp_configs[epoch] = ExperimentConfig(root_path=root_path,
                                              num_decoder_layer=4,
                                              z_dim=z_dim,
                                              num_units=num_units,
                                              num_cluster_config=num_cluster_config,
                                              confidence_decay_factor=5,
                                              beta=5,
                                              supervise_weight=150,
                                              dataset_name="mnist",
                                              split_name="Split_1",
                                              model_name="VAE",
                                              batch_size=64,
                                              eval_interval=300,
                                              name=experiment_name,
                                              num_val_samples=128,
                                              total_training_samples=60000,
                                              manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                              reconstruction_weight=1,
                                              activation_hidden_layer="RELU",
                                              activation_output_layer="SIGMOID"
                                              )

        exp_config = exp_configs[epoch]
        exp_config.check_and_create_directories(run_id)

        if "train" in data_set:
            filename = get_encoded_csv_file(exp_config, int(epoch), "train")
            train_df = pd.read_csv(os.path.join(exp_config.ANALYSIS_PATH, filename))
            training_accuracies.append(accuracy_score(train_df["label_predicted"].values, train_df["label"].values))
        if "val" in data_set:
            filename = get_encoded_csv_file(exp_config, int(epoch), "val")
            val_df = pd.read_csv(os.path.join(exp_config.ANALYSIS_PATH, filename))
            validation_accuracies.append(accuracy_score(val_df["label_predicted"].values, val_df["label"].values))
    epochs = list(exp_configs.keys())
    if "train" in data_set:
        plt.plot(epochs, training_accuracies, label="train_"+"z_dim_"+str(z_dim))
    if "val" in data_set:
        plt.plot(epochs, validation_accuracies, label="val_"+"z_dim_"+str(z_dim))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.title(f"Number of units {num_units}")
