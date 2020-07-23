import os

from analysis.annotation_utils import combine_annotation_sessions, combine_multiple_annotations
from analysis.annotation_utils import KEY_FOR_DATA_FRAME
from config import get_base_path, ExperimentConfig, check_and_create_folder

# Initialize variables
debug = False
annotator = "SUNIL"

N_3 = 32
N_2 = 128
N_1 = 64
z_dim = 10
run_id = 3
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
number_of_keys = 2
keys = [f"manual_annotation_set_{k+1}" for k in range(number_of_keys)]
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
# Verify if there is duplicate annotations for the same combination of ( batch, image_no, row_number_with_image )
data_dict = combine_multiple_annotations(data_dict, exp_config, run_id)

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

for key in keys:
    df = data_dict[key]["data_frame"]
    df = df[df["has_multiple_value"]]
    print(df.shape[0])