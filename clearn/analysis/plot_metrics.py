from typing import List
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao


def plot_z_dim_vs_accuracy(root_path: str,
                           experiment_name: str,
                           z_dim_range: List,
                           num_units: List,
                           num_cluster_config: str,
                           num_epochs: int,
                           run_id: int,
                           split_name: str,
                           num_val_samples: int,
                           activation_output_layer="SIGMOID",
                           dataset_name="mnist",
                           batch_size=64,
                           num_decoder_layer=4
                           ):
    dao = get_dao(dataset_name, split_name, num_val_samples)
    training_accuracies = []
    z_dims = []
    validation_accuracies = []
    for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
        exp_config = ExperimentConfig(root_path=root_path,
                                      num_decoder_layer=num_decoder_layer,
                                      z_dim=z_dim,
                                      num_units=num_units,
                                      num_cluster_config=num_cluster_config,
                                      confidence_decay_factor=5,
                                      beta=5,
                                      supervise_weight=1,
                                      dataset_name=dataset_name,
                                      split_name=split_name,
                                      model_name="VAE",
                                      batch_size=batch_size,
                                      name=experiment_name,
                                      num_val_samples=num_val_samples,
                                      total_training_samples=dao.number_of_training_samples,
                                      manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                      reconstruction_weight=1,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer=activation_output_layer
                                      )
        exp_config.check_and_create_directories(run_id)
        file_prefix = "/train_accuracy_*.csv"
        train_epochs, _train_accuracies = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        max_accuracy = np.max(_train_accuracies)
        max_index = np.argmax(_train_accuracies)
        training_accuracies.append(max_accuracy)

        exp_config.check_and_create_directories(run_id)
        file_prefix = "/val_accuracy_*.csv"
        val_epochs, _val_accuracies = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        max_accuracy = np.max(_val_accuracies)
        max_index = np.argmax(_val_accuracies)
        validation_accuracies.append(max_accuracy)

        z_dims.append(z_dim)

    plt.plot(z_dims, training_accuracies)
    plt.plot(z_dims, validation_accuracies)
    plt.legend(["train", "Validation"])
    plt.xlabel("z_dim")
    plt.ylabel("Accuracy")
    plt.title(f"Num units {num_units} Training epochs {num_epochs}")


def read_accuracy_from_file(file_prefix):
    df = None
    print(file_prefix)
    for file in glob.glob(file_prefix):
        print(file_prefix, file)
        temp_df = pd.read_csv(file)
        if df is None:
            df = temp_df
        else:
            pd.concat([df, temp_df], axis=1)
    return df


def plot_epoch_vs_metric(root_path: str,
                           experiment_name: str,
                           num_units: List[int],
                           num_cluster_config: str,
                           z_dim: int,
                           run_id: int,
                           dataset_types: List[str] = ["train", "test"],
                           activation_output_layer="SIGMOID",
                           dataset_name="mnist",
                           split_name="Split_1",
                           batch_size=64,
                           num_val_samples=128,
                           num_decoder_layer=4,
                           metrics: List[str]=["accuracy"],
                           legend_loc = "best"
                           ):
    colors = ['r','g','b','y']
    dao = get_dao(dataset_name, split_name, num_val_samples)
    exp_config = ExperimentConfig(root_path=root_path,
                                  num_decoder_layer=num_decoder_layer,
                                  z_dim=z_dim,
                                  num_units=num_units,
                                  num_cluster_config=num_cluster_config,
                                  confidence_decay_factor=5,
                                  beta=5,
                                  supervise_weight=1,
                                  dataset_name=dataset_name,
                                  split_name=split_name,
                                  model_name="VAE",
                                  batch_size=batch_size,
                                  name=experiment_name,
                                  num_val_samples=num_val_samples,
                                  total_training_samples=dao.number_of_training_samples,
                                  manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                  reconstruction_weight=1,
                                  activation_hidden_layer="RELU",
                                  activation_output_layer=activation_output_layer
                                  )
    exp_config.check_and_create_directories(run_id, False)
    file_prefix = "/metrics_*.csv"
    df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
    ax = [None] * len(metrics)
    plots = [None] * (len(metrics) * len(dataset_types))
    plot_number = 0
    for i, metric in enumerate(metrics):
      if i == 0:
            fig, main_axis = plt.subplots()
            ax[i] = main_axis
      else:
            ax[i] = main_axis.twinx()

      for dataset_type in dataset_types:
          plots[plot_number], = ax[i].plot(df["epoch"],
                   df[f"{dataset_type}_{metric}"],
                   colors[plot_number],
                   label=f"{dataset_type}_{metric}")
          plot_number += 1
      plt.ylabel(metric.title())
      plt.xlabel("Epochs")
    main_axis.legend(handles=plots, labels=[l.get_label() for l in plots], loc=legend_loc, shadow=True, fontsize='x-large')
    plt.title(f"Number of units {num_units} z_dim = {z_dim}")
    plt.grid()


def plot_hidden_units_accuracy_layerwise(root_path: str,
                                         experiment_name: str,
                                         num_units: List[List[int]],
                                         num_cluster_config: str,
                                         z_dim: int,
                                         run_id: int,
                                         dataset_types: List[str] = ["train", "test"],
                                         activation_output_layer="SIGMOID",
                                         dataset_name="mnist",
                                         split_name="Split_1",
                                         batch_size=64,
                                         num_val_samples=128,
                                         num_decoder_layer=4,
                                         layer_num=0,
                                         fixed_layers=[]
                                         ):
    dao = get_dao(dataset_name, split_name, num_val_samples)
    for dataset_name in dataset_types:
        plt.figure()

        plt.xlabel("Hidden Units")
        plt.ylabel("Max Accuracy")
        accuracies = dict()
        num_epochs_trained = -1
        for num_unit in num_units:
            if len(fixed_layers) > 0:
                skip_this = False
                for index, fixed_layer in enumerate(fixed_layers):
                    if num_unit[index] != fixed_layer:
                        skip_this = True
                        break
                if skip_this:
                    continue

            exp_config = ExperimentConfig(root_path=root_path,
                                          num_decoder_layer=num_decoder_layer,
                                          z_dim=z_dim,
                                          num_units=num_unit,
                                          num_cluster_config=num_cluster_config,
                                          confidence_decay_factor=5,
                                          beta=5,
                                          supervise_weight=1,
                                          dataset_name=dataset_name,
                                          split_name=split_name,
                                          model_name="VAE",
                                          batch_size=batch_size,
                                          name=experiment_name,
                                          num_val_samples=num_val_samples,
                                          total_training_samples=dao.number_of_training_samples,
                                          manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                          reconstruction_weight=1,
                                          activation_hidden_layer="RELU",
                                          activation_output_layer=activation_output_layer
                                          )
            exp_config.check_and_create_directories(run_id)

            file_prefix = "/accuracy_*.csv"
            df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
            print(df.shape, df[f"{dataset_name}_accuracy"].max())
            if num_unit[layer_num] in accuracies:
                accuracies[num_unit[layer_num]].append([sum(num_unit[layer_num + 1:]),
                                                        df[f"{dataset_name}_accuracy"].max()])
            else:
                accuracies[num_unit[layer_num]] = [[sum(num_unit[layer_num + 1:]),
                                                    df[f"{dataset_name}_accuracy"].max()]]

            _num_epochs_trained = df["epoch"].max() + 1
            if num_epochs_trained == -1:
                num_epochs_trained = _num_epochs_trained
            else:
                if _num_epochs_trained != num_epochs_trained:
                    print(f"Number of epochs for {num_unit} is {_num_epochs_trained}")

        for layer_0_units in accuracies.keys():
            x_y = np.asarray(accuracies[layer_0_units])
            plt.scatter(x_y[:, 0], x_y[:, 1], label=f"Units in layer {layer_num} {layer_0_units}")

        plt.legend(loc='lower right', shadow=True, fontsize='x-large')

        plt.title(f"Number of epochs trained {num_epochs_trained} Fixed units ={fixed_layers}")

    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.grid()
    return accuracies


def plot_accuracy_multiple_runs(root_path: str,
                                experiment_name: str,
                                num_units: List[int],
                                run_ids: List[int],
                                num_cluster_config: str,
                                z_dim: int,
                                activation_output_layer="SIGMOID",
                                dataset_name="mnist",
                                split_name="Split_1",
                                batch_size=64,
                                num_val_samples=128,
                                num_decoder_layer=4
                                ):
    dao = get_dao(dataset_name, split_name, num_val_samples)
    plt.figure()
    dataset_name = "test"
    plt.xlabel("Run_ID")
    plt.ylabel("Max Accuracy")
    accuracies = []
    num_epochs_trained = -1
    for run_id in run_ids:
        exp_config = ExperimentConfig(root_path=root_path,
                                      num_decoder_layer=num_decoder_layer,
                                      z_dim=z_dim,
                                      num_units=num_units,
                                      num_cluster_config=num_cluster_config,
                                      confidence_decay_factor=5,
                                      beta=5,
                                      supervise_weight=1,
                                      dataset_name=dataset_name,
                                      split_name=split_name,
                                      model_name="VAE",
                                      batch_size=batch_size,
                                      name=experiment_name,
                                      num_val_samples=num_val_samples,
                                      total_training_samples=dao.number_of_training_samples,
                                      manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                      reconstruction_weight=1,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer=activation_output_layer
                                      )
        exp_config.check_and_create_directories(run_id)

        file_prefix = "/accuracy_*.csv"
        df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        print(df.shape, df[f"{dataset_name}_accuracy"].max())
        accuracies.append(df[f"{dataset_name}_accuracy"].max())

        _num_epochs_trained = df["epoch"].max() + 1
        if num_epochs_trained == -1:
            num_epochs_trained = _num_epochs_trained
        else:
            if _num_epochs_trained != num_epochs_trained:
                print(f"Number of epochs for {run_id} is {_num_epochs_trained}")

    plt.plot(accuracies)

    return accuracies
