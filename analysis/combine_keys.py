import os
from collections import defaultdict

from analysis.annotation_utils import combine_annotation_sessions, combine_multiple_annotations
from analysis.annotation_utils import KEY_FOR_DATA_FRAME, get_combined_data_frame, get_combined_annotation
from analysis.annotation_utils import show_image_and_get_annotations
from config import get_base_path, ExperimentConfig, check_and_create_folder


# Initialize variables
debug = False
annotator = "SUNIL"

N_3 = 32
N_2 = 128
N_1 = 64
z_dim = 5
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
# keys = [f"manual_annotation_{k+1}" for k in range(number_of_keys)]
keys = ["manual_annotation_sunil",
        "manual_annotation_set_1",
        "manual_annotation_arya"
        ]
# keys = ["manual_annotation_corrected",
#         "manual_annotation_set_1",
#         "manual_annotation_set_2",
#         "manual_annotation_sunil"
#         ]
number_of_keys = len(keys)
num_val_images = 2
max_epoch = 5
num_rows = max_epoch * number_of_evaluation_per_epoch * NUMBER_OF_ROWS * num_val_images

# Read all the individual data frames into a dictionary of format {"annotator_id"}
base_path = get_base_path(exp_config.root_path,
                          exp_config.Z_DIM,
                          exp_config.num_units[2],
                          exp_config.num_units[1],
                          exp_config.num_cluster_config,
                          run_id=run_id
                          )
for key in keys:
    annotation_path = base_path + key
    if not os.listdir(annotation_path):
        print(f"No csv files found in directory. Skipping the directory")
        keys.remove(key)
data_dict = combine_annotation_sessions(keys=keys,
                                        base_path=base_path,
                                        max_epoch=max_epoch)
# Verify if there is duplicate annotations for the same combination of ( batch, image_no, row_number_with_image )
data_dict = combine_multiple_annotations(data_dict, exp_config, num_rows, run_id)
keys = [k for k in data_dict.keys()]
# Save the de-duped data frame
for key in keys:
    base_path = get_base_path(exp_config.root_path,
                              exp_config.Z_DIM,
                              exp_config.num_units[2],
                              exp_config.num_units[1],
                              exp_config.num_cluster_config,
                              run_id=run_id
                              )
    file_name = exp_config.get_annotation_result_path(base_path) + f"/{key}.csv"

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
        epoch_step_dict[_batch] = rows_to_annotate_all_images
        epoch_step_dict[_batch] = rows_to_annotate_all_images
print(rows_to_annotate_all_images)
#
# show the image and annotation for the rows
corrected_text_all_images = show_image_and_get_annotations(epoch_step_dict, exp_config)
# TODO save the annotation in json
#
# TODO update the data frame and save the results to output_csv
# TODO change this to insert
_df_combined["text"] = _df_combined[f"text_{keys[0]}"]
for _batch in epoch_step_dict.keys():
    epoch = int(_batch) // 935
    step = (int(_batch) % 935 // 300)
    rows_to_annotate_all_images = epoch_step_dict[_batch]
    for image_no in [0, 1]:
        corrected_text = corrected_text_all_images[_batch][image_no]
        num_rows_annotated = rows_to_annotate_all_images[image_no]
        _df_combined.loc[(_df_combined["epoch"] == epoch) & (_df_combined["step"] == step) & (_df_combined["_idx"] == image_no) & (_df_combined["num_rows_annotated"].isin(num_rows_annotated)), "text"] = corrected_text

file_name = exp_config.get_annotation_result_path(base_path) + "combined_corrected.csv"
print(file_name)
_df_combined.to_csv(file_name, index=False)
