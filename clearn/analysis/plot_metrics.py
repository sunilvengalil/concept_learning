from typing import List
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.utils.dir_utils import get_eval_result_dir


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
                           num_decoder_layer=4,
                           metric="accuracy",
                           cumulative_function=np.max
                           ):
    dao = get_dao(dataset_name, split_name, num_val_samples)
    training_accuracies = []
    z_dims = []
    validation_accuracies = []
    for z_dim in z_dim_range:
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
        file_prefix = f"/train_{metric}_*.csv"
        df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        if df is not None:
            _train_accuracies = df[metric].values
        else:
            # Try older version of accuracy file
            file_prefix = f"/{metric}_*.csv"
            df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
            if df is not None:
                if f"train_{metric}_mean" in df.columns:
                    _train_accuracies = df[f"train_{metric}_mean"].values
                    _val_accuracies = df[f"val_{metric}_mean"].values
                else:
                    _train_accuracies = df[f"train_{metric}"].values
                    _val_accuracies = df[f"val_{metric}"].values

            else:
                raise Exception(f"File does not exist {exp_config.ANALYSIS_PATH + file_prefix}")
        max_accuracy = cumulative_function(_train_accuracies)
        # max_index = np.argmax(_train_accuracies)
        training_accuracies.append(max_accuracy)

        exp_config.check_and_create_directories(run_id)

        file_prefix = f"/val_{metric}_*.csv"

        df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        if df is not None:
            _val_accuracies = df[metric].values
        else:
            # Try older version of accuracy file
            file_prefix = f"/{metric}_*.csv"
            df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
            if df is not None:
                if f"val_{metric}_mean" in df.columns:
                    _val_accuracies = df[f"val_{metric}_mean"].values
                else:
                    _val_accuracies = df[f"val_{metric}"].values
            else:
                raise Exception(f"File does not exist {exp_config.ANALYSIS_PATH + file_prefix}")
        max_accuracy = cumulative_function(_val_accuracies)
        # max_index = np.argmax(_val_accuracies)
        validation_accuracies.append(max_accuracy)

        z_dims.append(z_dim)

    plt.plot(z_dims, training_accuracies)
    plt.plot(z_dims, validation_accuracies)
    plt.legend(["train", "Validation"])
    plt.xlabel("z_dim")
    plt.ylabel(metric.capitalize())
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
                         metrics: List[str] = ["accuracy"],
                         legend_loc="best",
                         show_sample_images=True
                         ):
    colors = ['r', 'g', 'b', 'y']
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
    fig = plt.figure(figsize=[20, 10])
    for i, metric in enumerate(metrics):
        if i == 0:
            main_axis = plt.subplot(1, 2, 1)
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
    if show_sample_images:
        im_ax = plt.subplot(1, 2, 2)
        _num_epochs_trained = df["epoch"].max()
        _num_batches_train = dao.number_of_training_samples // exp_config.BATCH_SIZE

        reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                _num_epochs_trained,
                                                _num_batches_train)
        print(reconstructed_dir)
        sample_image = cv2.imread(reconstructed_dir + "/im_0.png")
        im_ax.imshow(sample_image)

    main_axis.legend(handles=plots,
                     labels=[l.get_label() for l in plots],
                     loc=legend_loc,
                     shadow=True,
                     fontsize='x-large')
    plt.suptitle(f"Number of units {num_units} z_dim = {z_dim}")
    plt.grid()


def plot_hidden_units_accuracy_layerwise(root_path: str,
                                         experiment_name: str,
                                         num_units: List[List[int]],
                                         num_cluster_config: str,
                                         z_dim: int,
                                         run_id: int,
                                         dataset_types: List[str] = ["train", "test"],
                                         dataset_name="mnist",
                                         split_name="Split_1",
                                         batch_size=64,
                                         num_val_samples=128,
                                         num_decoder_layer=4,
                                         layer_num=0,
                                         fixed_layers=[],
                                         metric="accuracy",
                                         cumulative_function="max"
                                         ):
    if cumulative_function == "max":
        function_to_cumulate = np.max
        arg_function = np.argmax
    elif cumulative_function == "min":
        function_to_cumulate = np.min
        arg_function = np.argmin
    else:
        raise Exception("Argument cumulative function should be either min or max")
    dao = get_dao(dataset_name, split_name, num_val_samples)
    for dataset_name in dataset_types:
        plt.figure(figsize=(18, 8))

        plt.xlabel("Hidden Units")
        plt.ylabel(f"Max Accuracy({dataset_name})")
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
                    print(f"Skipping num_units {num_unit}")
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
                                          activation_hidden_layer="RELU"
                                          )
            exp_config.check_and_create_directories(run_id, False)

            file_prefix = f"/{metric}*.csv"
            df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
            metric_col_name = f"{dataset_name}_{metric}_mean"
            if metric_col_name not in df.columns:
                metric_col_name = f"{dataset_name}_{metric}"

            metric_values = df[metric_col_name].values
            if num_unit[layer_num] in accuracies:
                accuracies[num_unit[layer_num]].append([sum(num_unit[layer_num + 1:]),
                                                        float(function_to_cumulate( metric_values)),
                                                        int(arg_function(metric_values))
                                                        ]
                                                       )
            else:
                accuracies[num_unit[layer_num]] = [[sum(num_unit[layer_num + 1:]),
                                                    float(function_to_cumulate(metric_values)),
                                                    int(arg_function(metric_values))
                                                    ]
                                                   ]
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
        plt.title(f"Number of epochs trained {num_epochs_trained}. Fixed units ={fixed_layers}")

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
        exp_config.check_and_create_directories(run_id, False)

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


def plot_epoch_vs_accuracy(root_path: str,
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
                           metric="accuracy",
                           legend_loc="best"
                           ):
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
    exp_config.check_and_create_directories(run_id)

    file_prefix = f"/{metric}_*.csv"
    print(exp_config.ANALYSIS_PATH + file_prefix)
    df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
    print(df.shape)
    for dataset_name in dataset_types:
        print(dataset_name)
        if f"{dataset_name}_{metric}_mean" in df.columns:
          plt.plot(df["epoch"], df[f"{dataset_name}_{metric}_mean"], label=f"{dataset_name}_z_dim_{z_dim}")
        else:
          plt.plot(df["epoch"], df[f"{dataset_name}_{metric}"], label=f"{dataset_name}_z_dim_{z_dim}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend(loc=legend_loc, shadow=True, fontsize='x-large')
    plt.title(f"Number of units {num_units}")
    plt.grid()
