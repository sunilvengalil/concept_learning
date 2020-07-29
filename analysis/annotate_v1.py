import csv
import cv2
import argparse
import os
from config import ExperimentConfig
import json
#annotator = "ARYA"
# annotator = "MANJU"
annotator = "SUNIL"
eval_interval = 300

N_3 = 32
N_2 = 128
N_1 = 64
Z_DIM = 10
run_id = 1
ROOT_PATH = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
exp_config = ExperimentConfig(ROOT_PATH,
                              4,
                              Z_DIM,
                              [N_1, N_2, N_3],
                              None
                              )
exp_config.check_and_create_directories(run_id)
BATCH_SIZE = exp_config.BATCH_SIZE
DATASET_NAME = exp_config.dataset_name
exp_config.check_and_create_directories(run_id, create=False)

NUMBER_OF_ROWS = 16
NUM_DIGITS_PER_ROW = 4
MAX_BACKUPS = 10
ANNOTATED_CSV = "annotation_correction.csv"
last_epoch = 50

ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation_combined"


# Initialize variables
counter_start = 2
idx_start = 0
idx_end = 298

counter = 1
batch_size = 64
num_samples = 60000 - 128
num_batches_per_epoch = num_samples // batch_size
number_of_evaluation_per_epoch = num_batches_per_epoch // eval_interval
num_eval_batches = 2
start_eval_batch = 0


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_eval_result_dir(result_path, _epoch=0, idx=0, orig_or_reconstructed="reconstructed"):
    orig_dir = check_folder(
        result_path + "/"
        + orig_or_reconstructed + '_{:02d}_{:04}/'.format(_epoch, idx))
    return orig_dir


def parse_args():
    desc = "Start annotation of images"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()


args = parse_args()
start_epoch = args.epoch
start_batch_id = args.batch
print(exp_config.PREDICTION_RESULTS_PATH)
# reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
#                                         start_epoch + 1,
#                                         start_batch_id + 1,
#                                         "reconstructed")
cv2.namedWindow("Image")
stop_annotation = False
left, top = (0, 0)
right, bottom = (222, 28)
width = right - left
height = bottom - top


annotation_csv = os.path.join(ANNOTATED_PATH, ANNOTATED_CSV)
if os.path.isfile(annotation_csv):
    num_backups = 0
    annotation_csv_backed_up = os.path.join(ANNOTATED_PATH,
                                            "corrected_backed_up_" + str(num_backups) + "_" + ANNOTATED_CSV)
    while os.path.isfile(annotation_csv_backed_up) and num_backups < MAX_BACKUPS:
        num_backups += 1
        annotation_csv_backed_up = os.path.join(ANNOTATED_PATH,
                                                "corrected_backed_up_" + str(num_backups) + "_" + ANNOTATED_CSV)

    print(annotation_csv, annotation_csv_backed_up)
    if num_backups == MAX_BACKUPS:
        raise Exception(f"{annotation_csv_backed_up} backups already exist. Please back up the "
                        f"required files manually and restart")
    annotation_csv = annotation_csv_backed_up

print(start_batch_id)
epoch = 2
step = 2
image_no = 0
batch = epoch * 935 + step * eval_interval
rows_to_annotate = [ 1,  4,  6, 11, 13, 14, 15, 16]
text_list = []
with open(annotation_csv, "a")as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["epoch", "step", "_idx", "num_rows_annotated", "text"])
    last_file_reached = False
    prev_file = None

    reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                            epoch,
                                            (step * eval_interval) - 1,
                                            "reconstructed")
    print(reconstructed_dir)
    _idx = image_no
    left, top = (0, 0)
    right, bottom = (222, 28)
    file = reconstructed_dir + "im_" + str(_idx) + ".png"
    if not os.path.isfile(file):
        raise Exception("File does not exist {}".format(file))

    im = cv2.imread(file)
    print(file)

    k = 0
    for num_rows_annotated in rows_to_annotate:
        image_to_show = im.copy()
        top = (num_rows_annotated - 1) * height
        bottom = top + height
        cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow("Image", image_to_show)

        print(str(num_rows_annotated ), end=':', flush=True)
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
        writer.writerow([epoch, step, _idx, num_rows_annotated, text])
        text_list.append(text)
    print(k, ord('q'))

print(json.dumps(text_list))
cv2.destroyAllWindows()
print("Annotation completed")
print(f"Saved results to {ANNOTATED_PATH}")
