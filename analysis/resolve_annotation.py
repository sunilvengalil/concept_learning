import json
import os
from collections import defaultdict

import pandas as pd
from analysis import CSV_COL_NAME_EPOCH, CSV_COL_NAME_IMAGE_ID,CSV_COL_NAME_ROW_ID_WITHIN_IMAGE,CSV_COL_NAME_STEP
from analysis.annotation_utils import get_annotations, get_combined_data_frame
from analysis.annotation_utils import show_image_and_get_annotations, get_corrections_for_de_duping, KEY_FOR_DATA_FRAME
from config import get_base_path, ExperimentConfig, check_and_create_folder
from utils.pandas_utils import get_combined_annotation, has_multiple_value, space_separated_string

# Initialize variables
debug = False
annotator = "SUNIL"
_eval_interval = 300

N_3 = 32
N_2 = 128
N_1 = 64
z_dim = 10
run_id = 1
num_label_files = 2

ROOT_PATH = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
exp_config = ExperimentConfig(ROOT_PATH,
                              4,
                              z_dim,
                              [N_1, N_2, N_3],
                              num_cluster_config=None
                              )
exp_config.check_and_create_directories(run_id)

NUMBER_OF_ROWS = 16
NUM_DIGITS_PER_ROW = 4

check_and_create_folder(exp_config.get_annotation_result_path())

# Setting debug = true will write all intermediate data frames
if debug:
    debug_path = os.path.join(exp_config.get_annotation_result_path(), "debug/")
    check_and_create_folder(debug_path)

num_samples = 60000 - 128
num_batches_per_epoch = num_samples // exp_config.BATCH_SIZE
number_of_evaluation_per_epoch = num_batches_per_epoch // exp_config.eval_interval

keys = ["manual_annotation_set_1", "manual_annotation_set_2"]
data_dict = dict()
max_epoch = 5

# Read all the individual data frames into a dictionary
for key in keys:
    base_path = get_base_path(ROOT_PATH,
                              z_dim,
                              N_3,
                              N_2,
                              exp_config.num_cluster_config,
                              run_id=run_id
                              )
    annotation_path = base_path + key
    df, _ = get_annotations(annotation_path, batches=None)
    df = df[df["epoch"] < max_epoch]
    # TODO Add code to fix invalid character in annotation
    # BASE_PATH = get_base_path(ROOT_PATH, z_dim, N_3, N_2, exp_config.num_cluster_config, run_id=run_id)
    # PREDICTION_RESULTS_PATH = os.path.join(BASE_PATH, "prediction_results/")
    # ANNOTATED_PATH = BASE_PATH + "manual_annotation"
    #
    # df, unique = get_annotations(ANNOTATED_PATH, batches=batch)
    # df = df.rename(columns={"text": f"text_run_id_{run_id}"})

    group_by_columns = [CSV_COL_NAME_EPOCH, CSV_COL_NAME_STEP, CSV_COL_NAME_IMAGE_ID, CSV_COL_NAME_ROW_ID_WITHIN_IMAGE]
    unique_df = df.groupby(group_by_columns).aggregate(lambda x: space_separated_string(x)).reset_index()
    distinct_values = unique_df.apply(lambda x: has_multiple_value("text", x), axis=1)
    unique_df.insert(loc=len(unique_df.columns),
                     column="has_multiple_value",
                     value=distinct_values
                     )
    unique_df = unique_df.rename(columns={"text": f"text_{key}"})
    data_dict[key] = {KEY_FOR_DATA_FRAME: unique_df}

# Verify if there is duplicate annotations for the same combination of (batch, image_no, row_number_with_image)

for key in data_dict.keys():
    base_path = get_base_path(ROOT_PATH, z_dim, N_3, N_2, exp_config.num_cluster_config, run_id=run_id)
    file_name = exp_config.get_annotation_result_path(base_path) + f"/{key}.csv"
    df = data_dict[key][KEY_FOR_DATA_FRAME]
    # See if de-duped file already exists. If yes just load it and skip to next data_frame the loop
    de_duped_file_name = os.path.join(file_name)
    if os.path.isfile(de_duped_file_name):
        df_de_duped = pd.read_csv(de_duped_file_name)
        continue

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
        corrected_text_all_images, batches_with_duplicate, rows_to_fix_for_duplicate, epoch_step_dict = get_corrections_for_de_duping(df, exp_config)
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

    if epoch_step_dict is None or len(epoch_step_dict) == 0:
        df.to_csv(file_name, index=False)
        continue

    # Update the corrected text in the data frame
    for _batch in epoch_step_dict.keys():
        epoch = int(_batch) // 935
        step = (int(_batch) % 935 // 300)
        for image_no in [0, 1]:
            column_name = f"text_{key}"
            num_rows_annotated = rows_to_fix_for_duplicate[image_no]
            corrected_text = corrected_text_all_images[_batch][image_no]
            df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (df["num_rows_annotated"].isin(num_rows_annotated)) & (df["_idx"] == image_no), column_name] = corrected_text
            df.loc[(df["has_multiple_value"]) & (df["epoch"] == epoch) & (df["step"] == step) & (df["num_rows_annotated"].isin(num_rows_annotated)) & (df["_idx"] == image_no), "has_multiple_value"] = False

    # put the de-duped data frame back into data_dict
    data_dict[key][KEY_FOR_DATA_FRAME] = df
    data_dict[key]["rows_to_fix_for_duplicate"] = rows_to_fix_for_duplicate
    data_dict[key]["batch_with_duplicate"] = batches_with_duplicate

    # Save the de-duped data frame
    data_dict[key][KEY_FOR_DATA_FRAME].to_csv(file_name, index=False)

# Combine annotations from multiple data frames into one
_df_combined = get_combined_data_frame(data_dict)
_df_combined.to_csv(exp_config.get_annotation_result_path() + "combined.csv", index=False)

col_names = [f"text_{_k}" for _k in keys]
annotation_same = _df_combined.apply(lambda x: get_combined_annotation(x, col_names), axis=1)

print(f"Number of rows with same annotation {sum(annotation_same)}")
_df_combined.insert(len(_df_combined.columns), column="annotations_same", value=annotation_same)


epoch_step_dict = defaultdict()
for _epoch in range(4):
    for _step in range(3):
        _batch = _epoch * 935 + _step * 300
        batch_filter = _df_combined["batch"] == _batch

        rows_to_annotate_all_images = list()

        for image_no in [0, 1]:
            filter_condition = (batch_filter & (_df_combined["annotations_same"] == False) & (_df_combined["_idx"] == image_no))
            rows_to_correct = _df_combined["num_rows_annotated"][filter_condition].values
            rows_to_annotate_all_images.append(rows_to_correct)

#         _rows_to_annotate_all_images = get_mismatching_rows(keys, _df_combined, _epoch, _step)
        epoch_step_dict[_batch] = rows_to_annotate_all_images
print(rows_to_annotate_all_images)
#
# # TODO get the image and annotation for the rows
corrected_text_all_images = show_image_and_get_annotations(epoch_step_dict, exp_config)
# TODO save the annotation in json
#
# # TODO update the data frame and save the results to output_csv
# TODO change this to insert
_df_combined["text"] = _df_combined[f"text_{keys[0]}"]
for _batch in epoch_step_dict.keys():
    epoch = int(_batch) // 935
    step = (int(_batch) % 935 // 300)
    rows_to_annotate_all_images = epoch_step_dict[_batch]
    for image_no in [0, 1]:
        corrected_text = corrected_text_all_images[_batch][image_no]
        num_rows_annotated = rows_to_annotate_all_images[image_no]
        _df_combined.loc[(_df_combined["epoch"] == epoch) & (_df_combined["step"] == step) & (_df_combined["_idx"]==image_no) & (_df_combined["num_rows_annotated"].isin(num_rows_annotated)), "text"] = corrected_text

file_name = exp_config.get_annotation_result_path(base_path) + "combined_corrected.csv"
print(file_name)
_df_combined.to_csv(file_name, index = False)
