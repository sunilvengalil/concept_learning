import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score
import os
import cv2
import json
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

from config import ExperimentConfig

from analysis import CSV_COL_NAME_EPOCH, CSV_COL_NAME_STEP, CSV_COL_NAME_IMAGE_ID, CSV_COL_NAME_ROW_ID_WITHIN_IMAGE
# import ROOT_PATH, z_dim, N_3, N_2, exp_config, run_id, max_epoch

from config import get_base_path
from utils.dir_utils import get_eval_result_dir
from utils.pandas_utils import space_separated_string, has_multiple_value

KEY_FOR_DATA_FRAME = "data_frame"


""" Return labels for validation set"""


def read_label(label_file, num_label_files):
    labels = {}
    for file_number in range(num_label_files):
        label_df = pd.read_csv(label_file.format(file_number))
        labels[file_number] = label_df["label"].values
    return labels


def get_label_reconstructed(_df, num_rows_per_image, num_digits_per_row):
    labels = np.ones(num_rows_per_image * num_digits_per_row) * -2
    _df = _df.fillna("xxxx")
    for row in _df.iterrows():
        text_ = [-1] * num_digits_per_row
        row_text_ = row[1]["text"]
        # print("row_text_", row_text_)
        if isinstance(row_text_, float):
            print("Converting float to string")
            row_text_ = str(row_text_)
        row_text_ = row_text_.strip()
        if len(row_text_) != 0:
            if len(row_text_) < 4:
                for i in range(4-len(row_text_)):
                    text_[i] = 0
            offset = 4 - len(row_text_)
            for i, c in enumerate(row_text_):
                if c.isdigit():
                    text_[i+offset] = int(c)
                elif c == 'x':
                    text_[i+offset] = -1
                else:
                    num_rows_annotated = row[1]["num_rows_annotated"]
                    raise Exception(f"Invalid character in annotated data - {num_rows_annotated} {row_text_} {c}")

        for i in range(num_digits_per_row):
            offset = (row[1]["num_rows_annotated"] - 1) * num_digits_per_row
            labels[i + offset] = text_[i]
    return labels


"""
If reviewed annotation exist get that, otherwise get un-reviewed annotation
"""


def get_annotations(annotated_path, batches=None):
    print("Reading annotation from ", annotated_path)
    if os.path.isfile(annotated_path + "/manual_annotation_corrected.csv"):
        # TODO check why f-string formatting is not working here
        # print(f"Loading annotation from {annotated_path}")
        df = pd.read_csv(os.path.join(annotated_path, "manual_annotation_corrected.csv"))
        unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
        df["epoch"] = df["epoch"].astype(int)
        df["step"] = df["step"].astype(int)
        df["_idx"] = df["_idx"].astype(int)
        df["num_rows_annotated"] = df["num_rows_annotated"].astype(int)
        df["batch"] = df["batch"].astype(int)
        print("Corrected annotation exists", df.shape)
    else:
        # print(f"Loading annotation from {annotated_path}")
        df = None
        for annotation_file in os.listdir(annotated_path):
            if annotation_file.rsplit(".", 1)[1] == "csv":
                annotation_csv = os.path.join(annotated_path, annotation_file)
                _df = pd.read_csv(annotation_csv)
                if df is None:
                    df = _df
                else:
                    df = pd.concat([df, _df])
        df = df.fillna("xxxx")
        unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
        df["epoch"] = df["epoch"].astype(int)
        df["step"] = df["step"].astype(int)
        df["_idx"] = df["_idx"].astype(int)
        df["num_rows_annotated"] = df["num_rows_annotated"].astype(int)
        df["batch"] = df["epoch"] * 935 + (df["step"] * 300)
        df["batch"] = df["batch"].astype(int)
    print("Read annotation ", df.shape)
    if batches is None:
        return df, unique
    else:
        df = df[df["batch"] == batches]
        unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
        return df, unique


def _compute_accuracy(df,
                      step,
                      epoch,
                      num_label_files,
                      labels,
                      num_rows_per_image,
                      num_digits_per_row
                      ):
    df1 = df[(df["epoch"] == epoch) & (df["step"] == step)]
    labels_batch = []
    reconstructed_batch = []
    for image_no in range(num_label_files):
        _df = df1[df1["_idx"] == image_no]
        if _df.shape[0] > 0:
            try:
                reconstructed = get_label_reconstructed(_df[["num_rows_annotated", "text"]],
                                                        num_rows_per_image,
                                                        num_digits_per_row)
            except Exception as e:
                print(f"Invalid character in annotation,epoch {epoch:01d} , step {step:01d}, image {image_no}")
                print(str(e))
                continue
            _reconstructed_indices = reconstructed != -2
            reconstructed_batch.extend(reconstructed[_reconstructed_indices])
            labels_batch.extend(labels[image_no][_reconstructed_indices])
    accuracy = accuracy_score(labels_batch, reconstructed_batch)
    return accuracy


def compute_accuracy(labels, gt_dir,
                     max_epoch,
                     num_label_files,
                     num_rows_per_image,
                     num_digits_per_row,
                     eval_interval,
                     corrected_annotation_file=None
                     ):
    df, unique = get_annotations(gt_dir, corrected_annotation_file)
    unique = unique[unique["count"] > 10]

    accuracies = []
    total_batches_finished = []
    for unique_combination in unique.iterrows():
        epoch = unique_combination[1]["epoch"]
        step = unique_combination[1]["step"]
        accuracy = _compute_accuracy(df,
                                     step,
                                     epoch,
                                     num_label_files,
                                     labels,
                                     num_rows_per_image,
                                     num_digits_per_row
                                     )
        accuracies.append(accuracy)
        total_batches_finished.append(epoch * 935 + step * eval_interval)
    accuracy_df = pd.DataFrame(
        {"Total_Batches_Finished": total_batches_finished, "Epochs": np.asarray(total_batches_finished) / 935,
         "Accuracy": accuracies})
    accuracy_df = accuracy_df[accuracy_df["Epochs"] < max_epoch]
    return accuracy_df


def get_images(pred_path, batches=None):
    if batches is None:
        return None
    epoch = 1 + batches // 935
    batch = (batches % 935) - 1
    reconstructed_path = os.path.join(pred_path, f"reconstructed_{epoch:02d}_{batch:04d}")
    images = {}
    for file in os.listdir(reconstructed_path):
        im = cv2.imread(os.path.join(reconstructed_path, file))
        key = file.rsplit(".", 1)[0]
        images[key] = im
    return images


def plot_reconstructed_image(images_dict, im_name):
    dpi = matplotlib.rcParams['figure.dpi']
    keys = [k for k in images_dict.keys()]
    im_1 = images_dict[keys[0]][im_name]
    im_2 = images_dict[keys[1]][im_name]

    height, width = im_1.shape[0], im_1.shape[1]
    figsize = 2 * 2 * width / float(dpi), 2 * height / float(dpi)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(im_name)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im_1, cmap="Greys")
    plt.title(keys[0])

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(im_2, cmap="Greys")
    plt.title(keys[1])

    fig.tight_layout()


def plot_reconstructed_image_with_label(images_dict, im_name):
    dpi = matplotlib.rcParams['figure.dpi']
    num_cols = 3
    keys = [k for k in images_dict.keys()]
    im_1 = images_dict[keys[0]][im_name]
    im_2 = images_dict[keys[1]][im_name]
    im_3 = images_dict[keys[2]]
    print(im_1.shape)

    height, width = im_1.shape[0], im_1.shape[1]
    figsize = num_cols * 2 * width / float(dpi), 2 * height / float(dpi)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(im_name)
    ax = fig.add_subplot(1, num_cols, 1)
    ax.imshow(im_1, cmap="Greys")
    plt.title(keys[0])

    ax = fig.add_subplot(1, num_cols, 2)
    ax.imshow(im_2, cmap="Greys")
    plt.title(keys[1])

    ax = fig.add_subplot(1, num_cols, 3)
    ax.imshow(im_3, cmap="Greys")
    plt.title(keys[2])

    fig.tight_layout()


def get_images_dict(keys, exp_config, epoch, step, run_id, eval_interval=300):
    batch = epoch * 935 + step * eval_interval
    images_for_batch = dict()
    for key in keys:
        base_path = get_base_path(exp_config.ROOT_PATH,
                                  exp_config.z_dim,
                                  exp_config.num_units[2],
                                  exp_config.num_units[1],
                                  exp_config.num_cluster_config,
                                  run_id=run_id)
        prediction_results_path = os.path.join(base_path,
                                               "prediction_results/")
        _images_dict = get_images(prediction_results_path,
                                  batches=batch)
        images_for_batch[key] = {"images": _images_dict}
    return images_for_batch


def get_combined_data_frame(data_dict):
    # first_key = get_first_key(data_dict)
    # if first_key is None or len(first_key) == 0:
    #     return None

    # iterable = iter(data_dict)
    # print(f"Shape of individual data frames {data_dict[first_key][KEY_FOR_DATA_FRAME].shape}")
    # df_combined = data_dict[next(iterable)][KEY_FOR_DATA_FRAME]

    df_combined = None
    for key in data_dict.keys():
        if df_combined is None:
            df_combined = data_dict[key][KEY_FOR_DATA_FRAME]
        else:
            df_combined = df_combined.merge(data_dict[key][KEY_FOR_DATA_FRAME],
                                                on=["epoch", "step", "_idx", "num_rows_annotated", "batch"])

    print(f"Combined into single data frame. Result shape {df_combined.shape}")
    return df_combined


def get_combined_annotation(row, column_names):
    text_0 = row[column_names[0]]
    all_same = True
    for column_name in column_names[1:]:
        all_same = all_same and (row[column_name] == text_0)
        if not all_same:
            return False
    return all_same


def get_mismatching_rows(keys: list,
                         df_combined: DataFrame,
                         epoch: int,
                         step: int,
                         eval_interval: int=300):

    batch = epoch * 935 + step * eval_interval
    # batch_filter = df_combined["batch"] == batch

    # print("Number of elements filtered by batch index", sum(batch_filter))
    # df_combined_for_batch = df_combined[batch_filter]

    # num_cols = len(df_combined_for_batch.columns)
    # df_combined_for_batch.insert(num_cols, "labels", labels_list)

    # print(f"Data frame shape for batch {df_combined_for_batch.shape}")
    col_names = [f"text_{_k}" for _k in keys]
    annotation_same = df_combined.apply(lambda x: get_combined_annotation(x, col_names),
                                                        axis=1)

    print(f"Number of rows with same annotation {sum(annotation_same)}")
    df_combined.insert(len(df_combined.columns), column="annotations_same", value=annotation_same)

    # filter_condition = (batch_filter & (df_combined["annotations_same"] == False) & (df_combined["_idx"] == image_no))
    filter_condition = ( (df_combined["annotations_same"]==False) )
    rows_to_annotate_all_images = list()

    for image_no in [0, 1]:
        rows_to_correct = df_combined["num_rows_annotated"][filter_condition & (df_combined["_idx"] == image_no)].values
        rows_to_annotate_all_images.append(rows_to_correct)

    # images_for_batch = get_images_dict(epoch, step, run_id)
    # df_combined_for_batch[df_combined_for_batch["annotations_same"] == False]
    return rows_to_annotate_all_images


def get_corrections_for_de_duping(df, exp_config):
    batches_with_duplicate = [[], []]
    rows_to_fix_for_duplicate = [None, None]

    for image_no in [0, 1]:
        filter_condition = (df["has_multiple_value"]) & (df["_idx"] == image_no)
        print(np.sum(filter_condition))
        batches_with_duplicate[image_no] = df["batch"][filter_condition].values.tolist()
        rows_to_fix_for_duplicate[image_no] = df["num_rows_annotated"][filter_condition].values.tolist()
        print(f"Number of batches with duplicate in image {image_no} {len(batches_with_duplicate[image_no])} ")
        print(f"Number of rows for image {image_no} {rows_to_fix_for_duplicate[image_no]}")

    if len(batches_with_duplicate[0]) + len(batches_with_duplicate[1]) == 0:
        return None, None, None, None

    epoch_step_dict = dict()
    for image_no in [0, 1]:
        rows_dict=defaultdict(list)
        for _batch, row in zip(batches_with_duplicate[image_no], rows_to_fix_for_duplicate[image_no]):
            rows_dict[_batch].append(row)

        # Initialize the dictionary
        for _batch in rows_dict.keys():
            epoch_step_dict[_batch] = [[], []]

        print("epoch_step_dict", epoch_step_dict)
        for _batch, rows in rows_dict.items():
            epoch_step_dict[_batch][image_no] = rows

    print(epoch_step_dict)

    corrected_text_all_images = show_image_and_get_annotations(epoch_step_dict, exp_config)
    return corrected_text_all_images, batches_with_duplicate, rows_to_fix_for_duplicate, epoch_step_dict


"""
Display the image to the user in an image window in opencv. 
Get the annotations for highlighted rows and return the result
@:param: epoch_step_dict:dict(batch_no:int, rows_to_annotate:list(int))
@:returns: Returns a dictionary (batch_no:int, corrected_text(list(str)))
"""


def show_image_and_get_annotations(epoch_step_dict, exp_config, start_eval_batch=0, end_eval_batch=2):
    corrected_text_all_images = defaultdict(list)

    for _batch in epoch_step_dict.keys():
        epoch = _batch // 935
        step = (_batch % 935 // 300)
        print(epoch, step)
        reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                epoch + 1,
                                                (step * exp_config.eval_interval) - 1)
        print(reconstructed_dir)
        for _idx in [0, 1]:
            rows_to_annotate = epoch_step_dict[_batch][_idx]
            left, top = (0, 0)
            right, bottom = (222, 28)
            height = bottom - top
            file = reconstructed_dir + "im_" + str(_idx) + ".png"
            if not os.path.isfile(file):
                raise Exception("File does not exist {}".format(file))

            im = cv2.imread(file)
            print(file)
            image_to_show = im.copy()
            cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.imshow("Image", image_to_show)
            k = 0
            text_list = []
            for num_rows_annotated in rows_to_annotate:
                image_to_show = im.copy()
                top = (num_rows_annotated - 1) * height
                bottom = top + height
                cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow("Image", image_to_show)
                print(str(num_rows_annotated), end=':', flush=True)
                text = ""
                k = 0
                # for each row
                while k != "\n":
                    k = cv2.waitKey(0)
                    if k == 13 or k == ord('q'):
                        break
                    k = chr(k)
                    if k != 113:
                        if len(text) < 4:
                            text = text + k
                            print(k, end='', flush=True)
                        elif k == 8:
                            text = text[:-1]
                            print("\nBack space pressed\n", text)
                            print(k, end='', flush=True)
                if len(text) == 0:
                    text = "xxxx"
                print(f"Full Text for row {num_rows_annotated:01d}:{text}")
                text_list.append(text)
                if k == ord('q'):
                    break
            corrected_text_all_images[_batch].append(text_list)
    return corrected_text_all_images


""" Read all the individual data frames from location  into a dictionary of format {"annotator_id"}"""


def combine_annotation_sessions(keys: list, base_path: str, max_epoch: int):
    data_dict = dict()
    for key in keys:
        annotation_path = base_path + key
        if not os.listdir(annotation_path):
            print(f"No csv files found in directory {annotation_path}")
            return data_dict
        df, _ = get_annotations(annotation_path, batches=None)
        df = df[df["epoch"] < max_epoch]
        # TODO Add code to fix invalid character in annotation
        # BASE_PATH = get_base_path(ROOT_PATH, z_dim, N_3, N_2, exp_config.num_cluster_config, run_id=run_id)
        # PREDICTION_RESULTS_PATH = os.path.join(BASE_PATH, "prediction_results/")
        # ANNOTATED_PATH = BASE_PATH + "manual_annotation"
        #
        # df, unique = get_annotations(ANNOTATED_PATH, batches=batch)
        # df = df.rename(columns={"text": f"text_run_id_{run_id}"})
        if "text" not in df.columns:
            print(f"Files in  {annotation_path} does not have a column called text")

        group_by_columns = [CSV_COL_NAME_EPOCH, CSV_COL_NAME_STEP, CSV_COL_NAME_IMAGE_ID,
                            CSV_COL_NAME_ROW_ID_WITHIN_IMAGE]
        unique_df = df.groupby(group_by_columns).aggregate(lambda x: space_separated_string(x)).reset_index()
        distinct_values = unique_df.apply(lambda x: has_multiple_value("text", x), axis=1)
        unique_df.insert(loc=len(unique_df.columns),
                         column="has_multiple_value",
                         value=distinct_values
                         )
        unique_df = unique_df.rename(columns={"text": f"text_{key}"})
        data_dict[key] = {KEY_FOR_DATA_FRAME: unique_df}
    return data_dict


""" Verify if there is duplicate annotations for the same combination of ( batch, image_no, row_number_with_image )"""


def combine_multiple_annotations(data_dict, exp_config, run_id):
    for key in data_dict.keys():
        base_path = get_base_path(exp_config.root_path,
                                  exp_config.Z_DIM,
                                  exp_config.num_units[2],
                                  exp_config.num_units[1],
                                  exp_config.num_cluster_config,
                                  run_id=run_id
                                  )
        annotation_path = base_path + key
        df = data_dict[key][KEY_FOR_DATA_FRAME]
        # See if manually de-duped file already exists. If yes go to the next annotator

        manually_de_duped_file = os.path.join(annotation_path, "manually_de_duped.json")
        if os.path.isfile(manually_de_duped_file):

            with open(manually_de_duped_file, "r") as json_file:
                manually_de_duped = json.load(json_file)

            if manually_de_duped is not None and len(manually_de_duped) > 0:
                corrected_text_all_images = manually_de_duped["corrected_text_all_images"]
                batches_with_duplicate = manually_de_duped["batches_with_duplicate"]
                rows_to_fix_for_duplicate = manually_de_duped["rows_to_fix_for_duplicate"]
                epoch_step_dict = manually_de_duped["epoch_step_dict"]
            else:
                epoch_step_dict = None
        else:
            corrected_text_all_images, batches_with_duplicate, rows_to_fix_for_duplicate, epoch_step_dict = get_corrections_for_de_duping(
                df, exp_config)
            # Save manually corrected results to a json file
            manually_de_duped = dict()
            if corrected_text_all_images is not None:
                manually_de_duped["corrected_text_all_images"] = corrected_text_all_images
            if batches_with_duplicate is not None:
                manually_de_duped["batches_with_duplicate"] = batches_with_duplicate
            if rows_to_fix_for_duplicate is not None:
                manually_de_duped["rows_to_fix_for_duplicate"] = rows_to_fix_for_duplicate
            # epoch_step_dict key is batch number  and value is list of list
            if epoch_step_dict is not None:
                manually_de_duped["epoch_step_dict"] = epoch_step_dict
            with open(manually_de_duped_file, "w") as json_file:
                json.dump(manually_de_duped, json_file)

        # If no manual correction, return the data_dict as it is
        if epoch_step_dict is None or len(epoch_step_dict) > 0:
            continue

        # Update the corrected text in the data frame
        for _batch in epoch_step_dict.keys():
            epoch = int(_batch) // 935
            step = (int(_batch) % 935 // 300)
            for image_no in [0, 1]:
                column_name = f"text_{key}"
                num_rows_annotated = rows_to_fix_for_duplicate[image_no]
                corrected_text = corrected_text_all_images[_batch][image_no]
                df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (
                df["num_rows_annotated"].isin(num_rows_annotated)) & (
                       df["_idx"] == image_no), column_name] = corrected_text
                df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (df["num_rows_annotated"].isin(num_rows_annotated)) & (
                       df["_idx"] == image_no), "has_multiple_value"] = False

        # put the de-duped data frame back into data_dict
        data_dict[key][KEY_FOR_DATA_FRAME] = df
        data_dict[key]["rows_to_fix_for_duplicate"] = rows_to_fix_for_duplicate
        data_dict[key]["batch_with_duplicate"] = batches_with_duplicate

        # Save the de-duped data frame
        # data_dict[key][KEY_FOR_DATA_FRAME].to_csv(file_name, index=False)

    return data_dict

