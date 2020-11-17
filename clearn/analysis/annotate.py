import csv
import cv2
import os
from clearn.utils.dir_utils import get_eval_result_dir
from clearn.utils.annotation_utils import ANNOTATION_FOLDER_NAME_PREFIX, annotator

eval_interval = 300


def annotate(exp_config, run_id, start_epoch, start_batch_id):
    exp_config.check_and_create_directories(run_id, create=False)
    number_of_rows = 16
    max_backups = 10
    annotated_csv = "annotation.csv"
    last_epoch = 50
    annotated_path = exp_config.BASE_PATH + ANNOTATION_FOLDER_NAME_PREFIX + str(annotator)

    # Initialize variables
    num_eval_batches = 2
    start_eval_batch = 0

    cv2.namedWindow("Image")
    stop_annotation = False
    if not os.path.isdir(annotated_path):
        os.mkdir(annotated_path)
    annotation_csv = os.path.join(annotated_path, annotated_csv)
    if os.path.isfile(annotation_csv):
        num_backups = 0
        annotation_csv_backed_up = os.path.join(annotated_path, "backed_up_" + str(num_backups) + "_"+annotated_csv)
        while os.path.isfile(annotation_csv_backed_up) and num_backups < max_backups:
            num_backups += 1
            annotation_csv_backed_up = os.path.join(annotated_path,
                                                    "backed_up_" + str(num_backups) + "_" + annotated_csv)
        print(annotation_csv, annotation_csv_backed_up)
        if num_backups == max_backups:
            raise Exception(f"{annotation_csv_backed_up} backups already exist. Please back up the "
                            f"required files manually and restart")
        annotation_csv = annotation_csv_backed_up

    with open(annotation_csv, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["epoch", "step", "_idx", "num_rows_annotated", "text"])
        last_file_reached = False
        prev_file = None
        for epoch in range(start_epoch, last_epoch):
            if epoch < start_epoch:
                continue
            for step in range(start_batch_id, 4):
                print(f"step:{step}")
                reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                        epoch,
                                                        (step * eval_interval),
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
                    while num_rows_annotated < number_of_rows and k != ord('q'):
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

                        top, left, bottom, right = highlight_next_row(top, left, bottom, right, im)
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
    print(f"Saved results to {annotated_path}")


def highlight_next_row(top, left, bottom, right, im):
    image_to_show = im.copy()
    height = bottom - top
    top += height
    bottom += height
    cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow("Image", image_to_show)
    return top, left, bottom, right
