import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import cv2
import json
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

from clearn.analysis import CSV_COL_NAME_EPOCH, CSV_COL_NAME_STEP, CSV_COL_NAME_IMAGE_ID
from clearn.analysis import CSV_COL_NAME_ROW_ID_WITHIN_IMAGE
from clearn.config import get_base_path, get_keys
from clearn.utils.dir_utils import get_eval_result_dir
from clearn.utils.pandas_utils import space_separated_string, has_multiple_value
from clearn.analysis.annotate_v3 import show_image_and_get_annotations_v2
from clearn.analysis import ANNOTATION_FOLDER_NAME_PREFIX

annotator = "sunil"

KEY_FOR_DATA_FRAME = "data_frame"


def read_label(label_file, num_label_files):
    """ Return labels for validation set"""
    labels = {}
    for file_number in range(num_label_files):
        label_df = pd.read_csv(label_file.format(file_number))
        labels[file_number] = label_df["label"].values
    return labels


def get_label_reconstructed(_df, num_rows_per_image, num_digits_per_row, text_column_name):
    labels = np.ones(num_rows_per_image * num_digits_per_row) * -2
    _df = _df.fillna("xxxx")
    for row in _df.iterrows():
        text_ = [-1] * num_digits_per_row
        row_text_ = row[1][text_column_name]
        # print("row_text_", row_text_)
        if isinstance(row_text_, float):
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


def get_labels_and_annotations(epoch, step, df_batch, text_column_name, num_label_files, labels):
    labels_batch = []
    reconstructed_batch = []
    for image_no in range(num_label_files):
        _df = df_batch[df_batch["_idx"] == image_no]

        if _df.shape[0] > 0:
            try:
                reconstructed = get_label_reconstructed(_df[["num_rows_annotated", text_column_name]],
                                                        num_rows_per_image=16,
                                                        num_digits_per_row=4,
                                                        text_column_name=text_column_name)
            except Exception as e:
                print(f"Invalid character in annotation,epoch {epoch:01d} , step {step:01d}, image {image_no}")
                print(str(e))
                continue
            _reconstructed_indices = reconstructed != -2
            reconstructed_batch.extend(reconstructed[_reconstructed_indices])
            labels_batch.extend(labels[image_no][_reconstructed_indices])
    return labels_batch, reconstructed_batch


def convert_data_type(df):
    df["epoch"] = df["epoch"].astype(int)
    df["step"] = df["step"].astype(int)
    df["_idx"] = df["_idx"].astype(int)
    df["num_rows_annotated"] = df["num_rows_annotated"].astype(int)
    df["batch"] = df["batch"].astype(int)


def get_combined_and_corrected_annotations(base_path,
                                           annotated_path,
                                           batches=None):
    print("Reading annotation from ", annotated_path)
    combined_and_corrected_annotation_file_name = "combined_corrected.csv"
    df = None
    if os.path.isfile(annotated_path + combined_and_corrected_annotation_file_name):
        df = pd.read_csv(os.path.join(annotated_path, combined_and_corrected_annotation_file_name))
    else:
        print("File does not exist", annotated_path + combined_and_corrected_annotation_file_name)

    # Intersection is empty so read individual annotatorID files
    keys = get_keys(base_path, ANNOTATION_FOLDER_NAME_PREFIX)
    for key in keys:
        if os.path.isfile(annotated_path + key + ".csv"):
            if df is None:
                df = pd.read_csv(os.path.join(annotated_path, key + ".csv"))
            else:
                _df = pd.read_csv(os.path.join(annotated_path, key + ".csv"))
                df = pd.concat([df, _df])
        else:
            print("File does not exist", annotated_path + combined_and_corrected_annotation_file_name)

    if batches is not None:
        df = df[df["batch"] == batches]

    unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
    convert_data_type(df)
    return df, unique


def get_annotations(annotated_path, batches=None):
    """
    If reviewed annotation exist get that, otherwise get un-reviewed annotation
    """
    print("Reading annotation from ", annotated_path)
    if os.path.isfile(annotated_path + "/manual_annotation_corrected.csv"):
        # TODO check why f-string formatting is not working here
        df = pd.read_csv(os.path.join(annotated_path, "manual_annotation_corrected.csv"))
        unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
        df["epoch"] = df["epoch"].astype(int)
        df["step"] = df["step"].astype(int)
        df["_idx"] = df["_idx"].astype(int)
        df["num_rows_annotated"] = df["num_rows_annotated"].astype(int)
        df["batch"] = df["batch"].astype(int)
    else:
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
        try:
            df["epoch"] = df["epoch"].astype(int)
            df["step"] = df["step"].astype(int)
            df["_idx"] = df["_idx"].astype(int)
            df["num_rows_annotated"] = df["num_rows_annotated"].astype(int)
            df["batch"] = df["epoch"] * 935 + (df["step"] * 300)
            df["batch"] = df["batch"].astype(int)
        except Exception as ex:
            print(f"Error while processing data in csv file located at {annotated_path}")
            print(ex)
            exit(0)
    if batches is None:
        return df, unique
    else:
        df = df[df["batch"] == batches]
        unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
        return df, unique


def _compute_accuracy(base_path,
                      df,
                      epoch,
                      step,
                      num_label_files,
                      labels
                      ):
    df_batch = df[(df["epoch"] == epoch) & (df["step"] == step)]
    print(df_batch.shape)
    print(epoch, step)
    keys = get_keys(base_path, ANNOTATION_FOLDER_NAME_PREFIX)
    text_columns = ["text"]
    text_columns.extend(["text_" + key for key in keys])
    print("text_columns", text_columns)
    for text_column_name in text_columns:
        if text_column_name not in df_batch.columns:
            continue
        number_of_nulls = df_batch[text_column_name].isnull().sum()
        print(text_column_name, number_of_nulls)
        if number_of_nulls > 0:
            print(f"skipping {text_column_name} as {text_column_name} values are null")
            continue
        labels_batch, reconstructed_batch = get_labels_and_annotations(epoch,
                                                                       step,
                                                                       df_batch,
                                                                       text_column_name,
                                                                       num_label_files,
                                                                       labels)
        print(f"Computed for  {text_column_name} for epoch {epoch} and step {step} ")
        break

    accuracy = accuracy_score(labels_batch, reconstructed_batch)
    return accuracy


def compute_accuracy(base_path,
                     labels,
                     gt_dir,
                     max_epoch,
                     num_label_files,
                     eval_interval,
                     corrected_annotation_file=None
                     ):
    df, unique = get_combined_and_corrected_annotations(base_path, gt_dir, corrected_annotation_file)
    accuracies = []
    total_batches_finished = []
    for unique_combination in unique.iterrows():
        epoch = unique_combination[1]["epoch"]
        step = unique_combination[1]["step"]
        accuracy = _compute_accuracy(base_path,
                                     df,
                                     epoch,
                                     step,
                                     num_label_files,
                                     labels
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
    height, width = im_1.shape[0], im_1.shape[1]
    figsize = 2 * len(keys) * width / float(dpi), 2 * height / float(dpi)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(im_name)

    for i in range(len(keys)):
        ax = fig.add_subplot(1, len(keys), i +1)
        ax.imshow(images_dict[keys[i]][im_name], cmap="Greys")
        plt.title(keys[i])
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
        base_path = get_base_path(exp_config,
                                  run_id=run_id)
        prediction_results_path = os.path.join(base_path,
                                               "prediction_results/")
        _images_dict = get_images(prediction_results_path,
                                  batches=batch)
        images_for_batch[key] = {"images": _images_dict}
    return images_for_batch

""" 
Join all dataframes in dictionary into a single dataframe
"""


def get_combined_data_frame(data_dict):
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
            return True
    return not all_same


def get_corrections_for_de_duping(df, exp_config, filter_column, manually_de_duped_file=None):
    epoch_step_dict = {}

    group_by_columns = [CSV_COL_NAME_EPOCH, CSV_COL_NAME_STEP]
    unique_df = df[group_by_columns].drop_duplicates()

    for index, row in unique_df.iterrows():
        _epoch = row.epoch
        _step = row.step
        _batch = _epoch * 935 + _step * 300
        batch_filter = df["batch"] == _batch
        rows_to_annotate_all_images = list()
        for image_no in [0, 1]:
            filter_condition = (batch_filter & (df[filter_column]) & (df["_idx"] == image_no))
            rows_to_correct = df["num_rows_annotated"][filter_condition].values.tolist()
            rows_to_annotate_all_images.append(rows_to_correct)
        # TODO do this for all validation images if there are more than two validation image
        if len(rows_to_annotate_all_images[0]) + len(rows_to_annotate_all_images[1]) > 0:
            epoch_step_dict[int(_batch)] = rows_to_annotate_all_images
    if manually_de_duped_file is None:
        corrected_text_all_images = show_image_and_get_annotations(epoch_step_dict, exp_config)
    else:
        print(epoch_step_dict)
        corrected_text_all_images = show_image_and_get_annotations_v2(epoch_step_dict,
                                                                      exp_config,
                                                                      manually_de_duped_file)
        print(corrected_text_all_images)
    return corrected_text_all_images, epoch_step_dict


"""
Display the image to the user in an image window in opencv. 
Get the annotations for highlighted rows and return the result
@:param: epoch_step_dict:dict(batch_no:int, rows_to_annotate:list(int))
@:returns: Returns a dictionary (batch_no:int, corrected_text(list(str)))
"""


def show_image_and_get_annotations(epoch_step_dict, exp_config):
    corrected_text_all_images = defaultdict(list)

    for _batch in epoch_step_dict.keys():
        epoch = _batch // 935
        step = (_batch % 935 // 300)
        reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                epoch,
                                                (step * exp_config.eval_interval))
        for _idx in [0, 1]:
            rows_to_annotate = epoch_step_dict[_batch][_idx]
            print(f"batch {_batch}  image {_idx}:{rows_to_annotate}")
            left, top = (0, 0)
            right, bottom = (222, 28)
            height = bottom - top
            file = reconstructed_dir + "im_" + str(_idx) + ".png"
            if not os.path.isfile(file):
                raise Exception("File does not exist {}".format(file))

            im = cv2.imread(file)
            text_list = []
            for num_rows_annotated in rows_to_annotate:
                image_to_show = im.copy()
                top = (num_rows_annotated - 1) * height
                bottom = top + height
                cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow("Image", image_to_show)
                cv2.waitKey(0)
                text = input(str(num_rows_annotated)+":")
                if len(text) == 0:
                    text = "xxxx"
                print(f"Full Text for row {num_rows_annotated:01d}:{text}")
                text_list.append(text)
                if text == 'q':
                    break
            corrected_text_all_images[_batch].append(text_list)
    return corrected_text_all_images


def get_annotations_for_keys(keys: dict, max_epoch: int, use_corrected=False):
    data_dict = dict()
    for key in keys:
        annotation_path = keys[key]
        if not os.listdir(annotation_path):
            print(f"No csv files found in directory {annotation_path}")
            return data_dict
        if use_corrected:
            df, _ = get_combined_and_corrected_annotations(annotation_path)
        else:
            df, _ = get_annotations(annotation_path, batches=None)
        df = df[df["epoch"] < max_epoch]
        data_dict[key] = {KEY_FOR_DATA_FRAME: df}
    return data_dict


def combine_annotation_sessions(keys: list, base_path: str, max_epoch: int):
    """ Read all the individual data frames from location `base_path`+key  into a dictionary of format {"annotator_id":dataframe}
    @:param keys List of keys- each key corresponds to annotation by a different user
    """
    data_dict = dict()
    for key in keys:
        annotation_path = base_path + key
        if not os.listdir(annotation_path):
            print(f"No csv files found in directory {annotation_path}")
            return data_dict
        df, _ = get_annotations(annotation_path, batches=None)
        df = df[df["epoch"] < max_epoch]
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
        print(f"Dataframe for key {key} {unique_df.shape} ")
    return data_dict


""" Verify if there is duplicate annotations for the same combination of ( batch, image_no, row_number_with_image )"""


def save_manually_de_duped_json(manually_de_duped_file,
                                corrected_text_all_images,
                                epoch_step_dict):
    # Save manually corrected results to a json file
    manually_de_duped = dict()
    if corrected_text_all_images is not None:
        manually_de_duped["corrected_text_all_images"] = corrected_text_all_images
        #manually_de_duped["corrected_text_all_images"] = {9015:"sunil"}
    # if batches_with_duplicate is not None:
    #     manually_de_duped["batches_with_duplicate"] = batches_with_duplicate
    # if rows_to_fix_for_duplicate is not None:
    #     manually_de_duped["rows_to_fix_for_duplicate"] = rows_to_fix_for_duplicate
    # epoch_step_dict key is batch number  and value is list of list
    if epoch_step_dict is not None:
        manually_de_duped["epoch_step_dict"] = epoch_step_dict
    with open(manually_de_duped_file, "w") as json_file:
        json.dump(manually_de_duped, json_file)
    return manually_de_duped


def get_manual_annotation(manually_de_duped_file, df, exp_config, filter_column):
    # See if manually de-duped file already exists. If yes go to the next annotator
    if os.path.isfile(manually_de_duped_file):
        with open(manually_de_duped_file, "r") as json_file:
            manually_de_duped = json.load(json_file)

        if manually_de_duped is not None and len(manually_de_duped) > 0:
            corrected_text_all_images = manually_de_duped["corrected_text_all_images"]
            epoch_step_dict = manually_de_duped["epoch_step_dict"]
        else:
            corrected_text_all_images = None
            epoch_step_dict = None
    else:
        corrected_text_all_images, epoch_step_dict = get_corrections_for_de_duping(
            df,
            exp_config,
            filter_column,
            manually_de_duped_file)
        print("Saving manually deduped json")
        save_manually_de_duped_json(manually_de_duped_file,
                                    corrected_text_all_images,
                                    epoch_step_dict)
    return corrected_text_all_images, epoch_step_dict


def combine_multiple_annotations(data_dict, exp_config, num_rows, run_id):
    keys_to_remove = []
    for key in data_dict.keys():
        base_path = get_base_path(exp_config,
                                  run_id=run_id
                                  )
        annotation_path = base_path + key
        df = data_dict[key][KEY_FOR_DATA_FRAME]

        manually_de_duped_file = os.path.join(annotation_path, "manually_de_duped.json")
        corrected_text_all_images, epoch_step_dict = get_manual_annotation(manually_de_duped_file,
                                                                           df,
                                                                           exp_config,
                                                                           "has_multiple_value")
        # If no manual correction, return the data_dict as it is
        if epoch_step_dict is None or len(epoch_step_dict) == 0:
            # if df.shape[0] != num_rows:
            #     keys_to_remove.append(key)
            continue

        # Update the corrected text in the data frame
        for _batch in epoch_step_dict.keys():
            epoch = int(_batch) // 935
            step = (int(_batch) % 935 // 300)
            for image_no in [0, 1]:
                column_name = f"text_{key}"
                num_rows_annotated = epoch_step_dict[_batch][image_no]
                corrected_text = corrected_text_all_images[_batch][image_no]
                df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (
                df["num_rows_annotated"].isin(num_rows_annotated)) & (
                       df["_idx"] == image_no), column_name] = corrected_text
                df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (df["num_rows_annotated"].isin(num_rows_annotated)) & (
                       df["_idx"] == image_no), "has_multiple_value"] = False

        # put the de-duped data frame back into data_dict
        # if df.shape[0] == num_rows:
        #    data_dict[key][KEY_FOR_DATA_FRAME] = df
        # else:
        #     print(f"Key {key} have {df.shape[0]} rows. Expected {num_rows} rows. Skipping this key")
        #     keys_to_remove.append(key)

        data_dict[key][KEY_FOR_DATA_FRAME] = df

    for key in keys_to_remove:
        del data_dict[key]

    return data_dict
