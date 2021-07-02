import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv

from clearn.analysis.plot_metrics import plot_hidden_units_accuracy_layerwise
from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.utils.dir_utils import get_eval_result_dir

def show_image_and_get_annotations_v3(epoch_step_dict,
                                      manually_de_duped_file
                                      ):
    cv2.namedWindow("Image")
    stop_annotation = False
    manually_de_duped_file = manually_de_duped_file.rsplit(".", 1)[0] + ".csv"
    print(manually_de_duped_file)

    annotation_csv = manually_de_duped_file
    corrected_text_all_images = {}
    step = 328

    with open(annotation_csv, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["epoch", "step", "_idx", "num_rows_annotated", "text"])
        prev_file = None
        for reconstructed_dir in epoch_step_dict.keys():
            # if _batch is None:
            #     continue
            # epoch = _batch // 935
            # step = (_batch % 935 // exp_config.eval_interval)
            # reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
            #                                         epoch,
            #                                         (step * exp_config.eval_interval),
            #                                         )
            # print(epoch, _batch, step, exp_config.eval_interval)
            print(reconstructed_dir)
            # key = int(_batch)
            _idx = 0
            for image_path in epoch_step_dict[reconstructed_dir]:
                text_list = []
                # print(f"batch {_batch}  image {_idx}:{rows_to_annotate}")
                left, top = (0, 0)
                right, bottom = (222, 28)
                height = bottom - top
                file = reconstructed_dir + "im_" + str(_idx) + ".png"

                if prev_file is None and not os.path.isfile(file):
                    raise Exception("File does not exist {}".format(file))
                if not os.path.isfile(file):
                    stop_annotation = True
                    break
                im = cv2.imread(image_path)
                print(image_path)
                for num_rows_annotated in range(16):
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
                    print("Full Text for row {:01d}:{}".format(num_rows_annotated, text))
                    _idx = _idx + 1
                    text_list.append(text)
                    writer.writerow([epoch, step, _idx, num_rows_annotated, text])
                if key in corrected_text_all_images:
                    corrected_text_all_images[key].append(text_list)
                else:
                    corrected_text_all_images[key] = [text_list]

                print(k, ord('q'))
                stop_annotation = k == ord('q')
                if stop_annotation:
                    break

                prev_file = file
            if stop_annotation:
                break

    cv2.destroyAllWindows()
    print("Annotation completed")
    print(f"Saved results to {manually_de_duped_file}")
    return corrected_text_all_images

def display_image(im):
    cv2.namedWindow("Image")
    stop_annotation = False
    corrected_text_all_images = {}
    step = 328
    text_list = []
    # print(f"batch {_batch}  image {_idx}:{rows_to_annotate}")
    left, top = (0, 0)
    right, bottom = (222, 28)
    height = bottom - top

    for num_rows_annotated in range(16):
        image_to_show = im.copy()
        top = num_rows_annotated * height
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
        print("Full Text for row {:01d}:{}".format(num_rows_annotated, text))
        text_list.append(text)
    return text_list


create_split = False
experiment_name = "find_architecture_unsup"
num_val_samples = -1
split_name ="Split_70_30"
reconstruction_weight = 1
dataset_name="mnist"
batch_size=128
z_dim_range = range(5, 30, 2)

num_epochs = 50
num_cluster_config = None
root_path="/Users/sunilv/concept_learning_exp"
run_id = 1
metric="reconstruction_loss"

num_units_list = [[64, 32],
                  [32, 32],
                  [16, 32],
                  [8, 32],
                  [4, 32],
                  [2, 32],
                      ]
num_unit_index = -1
batch_size = 128
dao = get_dao(dataset_name, split_name, num_val_samples)
z_dim = 11


def get_exp_config(num_units, z_dim):
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

"""
Get the images with best and worst performance
"""
with open("/Users/sunilv/concept_learning_exp/find_architecture_unsup/accuracy_vs_model_complexity.json") as filep:
    accuracies = json.load(filep)

# accuracies = plot_hidden_units_accuracy_layerwise(root_path=root_path,
#                         experiment_name=experiment_name,
#                         num_units=num_units_list[:-1],
#                         num_cluster_config=num_cluster_config,
#                         z_dim=z_dim,
#                         run_id=run_id,
#                         dataset_name=dataset_name,
#                         split_name=split_name,
#                         batch_size=batch_size,
#                         num_val_samples=num_val_samples,
#                         num_decoder_layer=5,
#                         layer_num=0,
#                         fixed_layers=[],
#                         dataset_types=["test"],
#                         metric=metric,
#                         cumulative_function="min"
#                       )
print(accuracies)


num_steps = 328
exp_config = get_exp_config(num_units_list[0], 11)
exp_config.check_and_create_directories(run_id, False)

# num_steps = dao.number_of_training_samples // batch_size
dataset_types = ["test", "val", "train"]
reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                        1,
                                        num_steps
                                        )

image_path = reconstructed_dir + dataset_types[0] + "_" + "TOP" + "_" + str(0) + ".png"
image = np.squeeze(cv2.imread(image_path))

separator = np.ones((image.shape[0], 10, image.shape[2]), np.uint8) * 255
large_separator = np.ones((image.shape[0], 30, image.shape[2]), np.uint8) * 255

fixed_layer = []
epoch_step_dict = dict()
for units_in_layer_0, x in accuracies.items():
    # print(x)
    for _x in x:
        num_units_after_layer_0 = _x[0]
        num_units = fixed_layer + [units_in_layer_0] + [num_units_after_layer_0]
        exp_config = get_exp_config(num_units, z_dim)
        exp_config.check_and_create_directories(run_id, False)
        reconstruction_loss = _x[1]
        min_epoch = _x[2]
        reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                min_epoch,
                                                num_steps
                                                )
        print(reconstructed_dir)
        for metrics_value in ["TOP"]:
            fig = plt.figure(figsize=(20, 10))
            plt.axis("off")
            # fig.tight_layout()
            images_for_dataset = dict()
            for dataset_type_name in dataset_types:
                images = [None] * 2
                for i in range(2):
                    image_path = reconstructed_dir + dataset_type_name + "_" + metrics_value + "_" + str(i) + ".png"
                    if reconstructed_dir in epoch_step_dict.keys():
                        epoch_step_dict[reconstructed_dir].append(image_path)
                    else:
                        epoch_step_dict[reconstructed_dir] = [image_path]
                    # print(image_path)
                    images[i] = np.squeeze(cv2.imread(image_path))
                    text_list = display_image(images[i])
                    print(text_list)
                # ax = fig.add_subplot(1, 2, i + 1)
                # print(separator.shape)
                combined_image = np.hstack((images[0], separator, images[1]))
                images_for_dataset[dataset_type_name] = combined_image

            final_image = images_for_dataset[dataset_types[0]]
            dataset_type_names_str = dataset_types[0]
            for dataset_type_name in dataset_types[1:]:
                final_image = np.hstack((final_image, large_separator, images_for_dataset[dataset_type_name]))
                dataset_type_names_str = dataset_type_names_str + ", " + dataset_type_name

            # plt.title(
            #     f"Reconstructed images. Number of units in hidden layer {units_in_layer_0}. Datasets {dataset_type_names_str} respectively from left to right ")
            # plt.imshow(final_image)

print(epoch_step_dict)
corrected_text_all_images = show_image_and_get_annotations_v3(epoch_step_dict)
print(corrected_text_all_images)
