from typing import List, Dict
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.utils.dir_utils import get_eval_result_dir

separator = np.ones((448, 10, 3), np.uint8) * 255
large_separator = np.ones((448, 30, 3), np.uint8) * 255


def plot_z_dim_vs_accuracy(root_path: str,
                           experiment_name: str,
                           z_dim_range: List,
                           num_units: List,
                           num_cluster_config: str,
                           num_epochs: int,
                           run_id: int,
                           split_name: str,
                           num_val_samples: int,
                           strides: List[int],
                           num_dense_layers: int,
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
                                      activation_output_layer=activation_output_layer,
                                      strides=strides,
                                      num_dense_layers=num_dense_layers
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
            df = pd.concat([df, temp_df], axis=0)
    return df


def plot_epoch_vs_metric(root_path: str,
                         experiment_name: str,
                         num_units: List[int],
                         num_cluster_config: str,
                         z_dim: int,
                         run_id: int,
                         strides,
                         num_dense_layers,
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
                                  activation_output_layer=activation_output_layer,
                                  strides=strides,
                                  num_dense_layers=num_dense_layers
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

            if f"{dataset_type}_{metric}_mean" in df.columns:
                metric_values = df[f"{dataset_type}_{metric}_mean"].values
            else:
                metric_values = df[f"{dataset_type}_{metric}"].values

            plots[plot_number], = ax[i].plot(df["epoch"],
                                             metric_values,
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


def get_exp_config(num_units,
                   z_dim,
                   root_path,
                   experiment_name,
                   batch_size,
                   dao,
                   num_cluster_config=None,
                   dataset_name="mnist",
                   split_name="Split_1",
                   num_val_samples=128
                   ):
    exp_config = ExperimentConfig(root_path=root_path,
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
                                  num_decoder_layer=2
                                  )
    return exp_config


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
                           strides:List[int],
                           num_dense_layers:int,
                           dataset_types: List[str] = ["train", "test"],
                           activation_output_layer="SIGMOID",
                           dataset_name="mnist",
                           split_name="Split_1",
                           batch_size=64,
                           num_val_samples=128,
                           num_decoder_layer=4,
                           metric="accuracy",
                           legend_loc="best",
                           exp_config=None,
                           confidence=False,
                           max_epoch=100,
                           max_accuracy=-1,
                           min_accuracy=0,
                           plot_filename=None
                           ):
    axis_font = {'fontname':'Arial', 'size':'26', "fontweight":"bold"}

    dao = get_dao(dataset_name, split_name, num_val_samples)
    if exp_config is None:
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
                                      activation_output_layer=activation_output_layer,
                                      strides=strides,
                                      num_dense_layers=num_dense_layers
                                      )

    if not exp_config.check_and_create_directories(run_id, False):
        raise Exception(" Result directories does not exist")

    file_prefix = f"/{metric}_*.csv"
    plt.figure(figsize=(16, 9))

    print(exp_config.ANALYSIS_PATH + file_prefix)
    if os.path.isfile(exp_config.ANALYSIS_PATH + file_prefix):
        df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        df = df[df["epoch"] < max_epoch]
        for dataset_name in dataset_types:
            print(dataset_name)
            if f"{dataset_name}_{metric}_mean" in df.columns:
                metric_values = df[f"{dataset_name}_{metric}_mean"]
            else:
                metric_values = df[f"{dataset_name}_{metric}"]
    else:
        file_prefix = "/metrics_*.csv"
        df = read_accuracy_from_file(exp_config.ANALYSIS_PATH + file_prefix)
        df = df[df["epoch"] < max_epoch]
        for dataset_type in dataset_types:
            if confidence:
                metric_values = df[f"{dataset_type}_{metric}_std"].values
            else:
                if f"{dataset_type}_{metric}_mean" in df.columns:
                    metric_values = df[f"{dataset_type}_{metric}_mean"].values
                else:
                    metric_values = df[f"{dataset_type}_{metric}"].values

    plt.plot(df["epoch"], metric_values, label=f"{dataset_type}", lw=2)

    plt.xlabel("Epochs", **axis_font)
    plt.ylabel(metric.capitalize(), **axis_font)
    if max_accuracy != -1:
        plt.yticks(ticks = [i for i in range(min_accuracy, max_accuracy, max_accuracy // 10)],
                   labels=[i for i in range(min_accuracy, max_accuracy, max_accuracy // 10)],
                   **axis_font)
    plt.xticks(**axis_font)
    plt.legend(loc=legend_loc, shadow=True, fontsize='x-large')
    plt.title(f"Number of units {num_units}")
    plt.grid(axis="x")
    if plot_filename is not None:
        plt.savefig(os.path.join(exp_config.ANALYSIS_PATH + "/"+ plot_filename), bbox="tight")
    return metric_values


def plot_hidden_units_accuracy_layerwise(root_path: str,
                                         experiment_name: str,
                                         num_units: List[List[int]],
                                         num_cluster_config: str,
                                         z_dim: int,
                                         run_id: int,
                                         strides:List[int],
                                         num_dense_layers:int,
                                         dataset_types: List[str] = ["train", "test"],
                                         dataset_name="mnist",
                                         split_name="Split_1",
                                         batch_size=64,
                                         num_val_samples=128,
                                         num_decoder_layer=4,
                                         layer_num=0,
                                         fixed_layers=[],
                                         metric="accuracy",
                                         cumulative_function="max",
                                         fig_size=(20,8),
                                         fname=None,
                                         accuracies:Dict[int, List]=None,
                                         legend_lc="best"
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
        plt.figure(figsize=fig_size)

        plt.xlabel("Hidden Units")
        plt.ylabel(f"Max {metric.capitalize()}({dataset_name})")
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
                                          activation_hidden_layer="RELU",
                                          strides=strides,
                                          num_dense_layers=num_dense_layers
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
            sns.scatterplot(x_y[:, 0], x_y[:, 1], label=f"Units in layer {layer_num} {layer_0_units}", s=450)
            sns.lineplot(x_y[:, 0], x_y[:, 1])
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.title(f"Dataset:{dataset_name} Number of epochs trained {num_epochs_trained}. Fixed units ={fixed_layers}")
        if fname is not None:
            fname = exp_config.ANALYSIS_PATH + fname
            print(f"Saving plot in file {fname}")
            plt.savefig(fname, bbox="tight")

    plt.legend(loc=legend_lc, shadow=True, fontsize='x-large')
    plt.grid()
    return accuracies


def display_reconstructed_images(exp_config: ExperimentConfig,
                                 dao,
                                 run_id,
                                 accuracies,
                                 dataset_types,
                                 num_steps,
                                 top_or_bottom,
                                 fname: str):
    fixed_layer = []
    print(fname)
    for units_in_layer_0, x in accuracies.items():
        for _x in x:
            num_units_after_layer_0 = _x[0]
            num_units = fixed_layer + [units_in_layer_0] + [num_units_after_layer_0]
            exp_config = get_exp_config(num_units,
                                        exp_config.Z_DIM,
                                        exp_config.root_path,
                                        exp_config.name,
                                        exp_config.BATCH_SIZE,
                                        dao
                                        )
            exp_config.check_and_create_directories(run_id, False)
            reconstruction_loss = _x[1]
            min_epoch = _x[2]
            reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                    min_epoch,
                                                    num_steps
                                                    )
            for metrics_value in [top_or_bottom]:
                fig = plt.figure(figsize=(20, 10))
                plt.axis("off")
                images_for_dataset = dict()
                for dataset_type_name in dataset_types:
                    images = [None] * 2
                    for i in range(2):
                        image_path = reconstructed_dir + dataset_type_name + "_" + metrics_value + "_" + str(i) + ".png"
                        images[i] = np.squeeze(cv2.imread(image_path))
                    combined_image = np.hstack((images[0], separator, images[1]))
                    images_for_dataset[dataset_type_name] = combined_image

                final_image = images_for_dataset[dataset_types[0]]
                dataset_type_names_str = dataset_types[0]
                for dataset_type_name in dataset_types[1:]:
                    final_image = np.hstack((final_image, large_separator, images_for_dataset[dataset_type_name]))
                    dataset_type_names_str = dataset_type_names_str + ", " + dataset_type_name
                plt.title(
                    f"Reconstructed images. Number of units in hidden layer {units_in_layer_0}. Datasets {dataset_type_names_str} respectively from left to right. Loss {reconstruction_loss} ")
                plt.imshow(final_image)
                print(fname)
                if fname is not None:
                    fqfn = f"{exp_config.root_path}/{exp_config.name}/{str(units_in_layer_0)}_{metrics_value}_{fname}"
                    print(f"Saving file at")
                    print(fqfn)
                    plt.margins(0, 0)
                    plt.savefig(fqfn + ".pdf", bbox="tight")
                    plt.savefig(fqfn + "jpg", bbox="tight")


def display_reconstructed_images_loss_wise(exp_config: ExperimentConfig,
                                 run_id,
                                 accuracies,
                                 dataset_type_name:str,
                                 num_steps,
                                 fname: str):
    fixed_layer = []
    print(fname)
    for units_in_layer_0, x in accuracies.items():
        for _x in x:
            num_units_after_layer_0 = _x[0]
            num_units = fixed_layer + [units_in_layer_0] + [num_units_after_layer_0]
            exp_config = get_exp_config(num_units,
                                        exp_config.Z_DIM,
                                        exp_config.root_path,
                                        exp_config.name,
                                        exp_config.BATCH_SIZE,
                                        dao
                                        )
            exp_config.check_and_create_directories(run_id, False)
            reconstruction_loss = _x[1]
            min_epoch = _x[2]
            reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                    min_epoch,
                                                    num_steps
                                                    )
            image_no = 0
            combined_image = None
            for metrics_value in ["TOP", "BOTTOM"]:
                fig = plt.figure(figsize=(20, 10))
                plt.axis("off")

                image_path = reconstructed_dir + dataset_type_name + "_" + metrics_value + "_" + str(image_no) + ".png"
                image = np.squeeze(cv2.imread(image_path))
                if combined_image is None:
                    combined_image = image
                else:
                    combined_image = np.hstack([combined_image, separator, image])
                plt.title(
                    f"Reconstructed images. Number of units in hidden layer {units_in_layer_0}. Datasets {dataset_type_name}. Loss {reconstruction_loss} ")
                plt.imshow(combined_image)
                print(fname)
                if fname is not None:
                    fqfn = f"{exp_config.root_path}/{exp_config.name}/{str(units_in_layer_0)}_{metrics_value}_{fname}"
                    print(f"Saving file at")
                    print(fqfn)
                    plt.margins(0, 0)
                    plt.savefig(fqfn + ".pdf", bbox="tight")
                    plt.savefig(fqfn + "jpg", bbox="tight")

