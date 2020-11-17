import os
from clearn.utils.annotation_utils import combine_annotation_sessions, combine_multiple_annotations
from clearn.utils.annotation_utils import KEY_FOR_DATA_FRAME, get_combined_data_frame
from clearn.utils.annotation_utils import get_manual_annotation, get_combined_annotation
from clearn.config import get_base_path, check_and_create_folder, get_keys
from clearn.analysis import ANNOTATION_FOLDER_NAME_PREFIX

# Initialize variables
def combine_keys(exp_config, run_id):
    debug = False
    exp_config.check_and_create_directories(run_id)

    NUMBER_OF_ROWS = 16
    check_and_create_folder(exp_config.get_annotation_result_path())

    # Setting debug = true will write all intermediate data frames
    if debug:
        debug_path = os.path.join(exp_config.get_annotation_result_path(), "debug/")
        check_and_create_folder(debug_path)

    num_batches_per_epoch = exp_config.num_train_samples // exp_config.BATCH_SIZE
    number_of_evaluation_per_epoch = num_batches_per_epoch // exp_config.eval_interval

    num_val_images = 2
    max_epoch = 20
    num_rows = max_epoch * number_of_evaluation_per_epoch * NUMBER_OF_ROWS * num_val_images

    # Read all the individual data frames into a dictionary of format {"annotator_id"}
    base_path = get_base_path(exp_config, run_id=run_id)

    keys = get_keys(base_path, ANNOTATION_FOLDER_NAME_PREFIX)
    print("keys", keys)

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
        base_path = get_base_path(exp_config,
                                  run_id=run_id
                                  )
        file_name = exp_config.get_annotation_result_path(base_path) + f"/{key}.csv"

        data_dict[key][KEY_FOR_DATA_FRAME].to_csv(file_name, index=False)

    print("******Completed de-duping*******")
    # Combine annotations from multiple data frames into one
    _df_combined = get_combined_data_frame(data_dict)
    _df_combined.to_csv(exp_config.get_annotation_result_path() + "combined.csv", index=False)
    col_names = [f"text_{_k}" for _k in keys]
    annotation_different = _df_combined.apply(lambda x: get_combined_annotation(x, col_names), axis=1)

    # TODO fix this for run_id = 0
    print(f"Number of rows with different annotation {sum(annotation_different)}")
    _df_combined.insert(len(_df_combined.columns), column="annotations_different", value=annotation_different)

    #
    # show the image and annotation for the rows
    manually_de_duped_file = exp_config.get_annotation_result_path() + "manually_de_duped.json"
    corrected_text_all_images, epoch_step_dict = get_manual_annotation(manually_de_duped_file,
                                                                       _df_combined,
                                                                       exp_config,
                                                                       "annotations_different")
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
    print("Writing final result to ", file_name)
    _df_combined.to_csv(file_name, index=False)
