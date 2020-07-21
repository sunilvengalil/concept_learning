import csv
import cv2
import argparse
import os
from config import ExperimentConfig
from utils.dir_utils import get_eval_result_dir

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
BATCH_SIZE = exp_config.BATCH_SIZE
DATASET_NAME = exp_config.dataset_name
exp_config.check_and_create_directories(run_id, create=False)

NUMBER_OF_ROWS = 16
NUM_DIGITS_PER_ROW = 4
MAX_BACKUPS = 10
ANNOTATED_CSV = "annotation.csv"
last_epoch = 50

ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation"
if annotator == "SUNIL":
    ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation_sunil"
elif annotator == "MANJU":
    BASE_PATH = "/home/test/"
    ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation_manju"
elif annotator == "ARYA":
    ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation_arya"
else:
    raise Exception("Only two annotators {} and {} are allowed now".format("SUNIL", "VIJAY"))

ANNOTATED_PATH = exp_config.BASE_PATH + "manual_annotation_set1"

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
reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                        start_epoch + 1,
                                        start_batch_id + 1)
cv2.namedWindow("Image")
stop_annotation = False
left, top = (0, 0)
right, bottom = (222, 28)
width = right - left
height = bottom - top

print(ANNOTATED_PATH)
if not os.path.isdir(ANNOTATED_PATH):
    os.mkdir(ANNOTATED_PATH)
annotation_csv = os.path.join(ANNOTATED_PATH, ANNOTATED_CSV)
if os.path.isfile(annotation_csv):
    num_backups = 0
    annotation_csv_backed_up = os.path.join(ANNOTATED_PATH, "backed_up_" + str(num_backups) + "_"+ANNOTATED_CSV)
    while os.path.isfile(annotation_csv_backed_up) and num_backups < MAX_BACKUPS:
        num_backups += 1
        annotation_csv_backed_up = os.path.join(ANNOTATED_PATH, "backed_up_" + str(num_backups) + "_" + ANNOTATED_CSV)

    print(annotation_csv, annotation_csv_backed_up)
    if num_backups == MAX_BACKUPS:
        raise Exception(f"{annotation_csv_backed_up} backups already exist. Please back up the "
                        f"required files manually and restart")
    annotation_csv = annotation_csv_backed_up
print(start_batch_id)
with open(annotation_csv, "a")as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["epoch", "step", "_idx", "num_rows_annotated", "text"])
    last_file_reached = False
    prev_file = None
    for epoch in range(start_epoch, last_epoch):
        if epoch < start_epoch:
            continue
        for step in range(start_batch_id, 4):
            print(step)
            reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                    epoch+1,
                                                    (step * eval_interval) - 1,
                                                    )
            print(reconstructed_dir)
            for _idx in range(start_eval_batch, num_eval_batches):
                left, top = (0, 0)
                right, bottom = (222, 28)
                file = reconstructed_dir + "im_" + str(_idx) + ".png"
                if prev_file is None and not os.path.isfile(file):
                    raise Exception("File does not exist {}".format(file))
                if not os.path.isfile(file):
                    last_file_reached = True
                    stop_annotation = True
                    break
                im = cv2.imread(file)
                print(file)
                image_to_show = im.copy()
                cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow("Image", image_to_show)
                num_rows_annotated = 0
                k = 0
                while num_rows_annotated < NUMBER_OF_ROWS and k != ord('q'):
                    print(str(num_rows_annotated+1), end=':', flush=True)
                    text = ""
                    k = 0
                    # one_row_completed = False
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
                    writer.writerow([epoch, step, _idx, num_rows_annotated + 1, text])
                    num_rows_annotated += 1

                    image_to_show = im.copy()
                    top += height
                    bottom += height
                    cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.imshow("Image", image_to_show)
                print(k, ord('q'))
                stop_annotation = k == ord('q')
                if stop_annotation:
                    break

                prev_file = file
            if stop_annotation:
                break
        if stop_annotation:
            break

if last_file_reached:
    print("Last file reached")
cv2.destroyAllWindows()
print("Annotation completed")
print(f"Saved results to {ANNOTATED_PATH}")
