import json
import os

import pandas as pd

from analysis.annotation_utils import combine_annotation_sessions
from analysis.annotation_utils import get_corrections_for_de_duping, KEY_FOR_DATA_FRAME
from config import get_base_path, ExperimentConfig, check_and_create_folder

# Initialize variables
debug = False
annotator = "SUNIL"

N_3 = 32
N_2 = 128
N_1 = 64
z_dim = 10
run_id = 1
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

num_batches_per_epoch = exp_config.num_train_samples // exp_config.BATCH_SIZE
number_of_evaluation_per_epoch = num_batches_per_epoch // exp_config.eval_interval

# keys = [f"annotated_by_{annotator_id}" for annotator_id in annotators()]
keys=["manual_annotation_set_1", "manual_annotation_set_2"]
max_epoch = 5

# Read all the individual data frames into a dictionary of format {"annotator_id"}
base_path = get_base_path(exp_config.root_path,
                          exp_config.Z_DIM,
                          exp_config.num_units[2],
                          exp_config.num_units[1],
                          exp_config.num_cluster_config,
                          run_id=run_id
                          )
data_dict = combine_annotation_sessions(keys=keys,
                                        base_path = base_path,
                                        max_epoch=max_epoch)

df = data_dict["manual_annotation_set_1"]["data_frame"]
df = df[df["has_multiple_value"]]
print(df.shape)

df = data_dict["manual_annotation_set_2"]["data_frame"]
df = df[df["has_multiple_value"]]
print(df.shape)

# Verify if there is duplicate annotations for the same combination of ( batch, image_no, row_number_with_image )
for key in data_dict.keys():
    base_path = get_base_path(exp_config.root_path,
                              exp_config.Z_DIM,
                              exp_config.num_units[2],
                              exp_config.num_units[1],
                              exp_config.num_cluster_config,
                              run_id=run_id
                              )
    annotation_path = base_path + key
    file_name = exp_config.get_annotation_result_path(base_path) + f"/{key}.csv"
    df = data_dict[key][KEY_FOR_DATA_FRAME]
    # See if manually de-duped file already exists. If yes go to the next annotator

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
