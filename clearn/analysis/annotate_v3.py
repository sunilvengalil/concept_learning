import csv
import cv2
import os
from clearn.utils.dir_utils import get_eval_result_dir
# annotator = "arya"
# annotator = "manju"
eval_interval = 300


def show_image_and_get_annotations_v2(epoch_step_dict,
                                      exp_config,
                                      manually_de_duped_file
                                      ):
    cv2.namedWindow("Image")
    stop_annotation = False
    manually_de_duped_file = manually_de_duped_file.rsplit(".", 1)[0] + ".csv"
    print(manually_de_duped_file)

    annotation_csv = manually_de_duped_file

    with open(annotation_csv, "a")as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["epoch", "step", "_idx", "num_rows_annotated", "text"])
        prev_file = None
        for _batch in epoch_step_dict.keys():
            epoch = _batch // 935
            step = (_batch % 935 // 300)
            reconstructed_dir = get_eval_result_dir(exp_config.PREDICTION_RESULTS_PATH,
                                                    epoch+1,
                                                    (step * eval_interval),
                                                    )
            print(reconstructed_dir)
            for _idx in [0, 1]:
                rows_to_annotate = epoch_step_dict[_batch][_idx]
                if len(rows_to_annotate) == 0:
                    continue
                print(f"batch {_batch}  image {_idx}:{rows_to_annotate}")
                left, top = (0, 0)
                right, bottom = (222, 28)
                height = bottom - top
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
                top = (rows_to_annotate[0] - 1) * height
                bottom = top + height

                cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.imshow("Image", image_to_show)
                k = cv2.waitKey(0)
                for num_rows_annotated in rows_to_annotate[1:]:
                    print(str(num_rows_annotated + 1), end=':', flush=True)
                    text = ""
                    k = 0
                    # for each row
                    while k != "\n":
                        # print("Waiting for key press")
                        # k = cv2.waitKey(0)
                        print(k)
                        exit()
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

    if last_file_reached:
        print("Last file reached")
    cv2.destroyAllWindows()
    print("Annotation completed")
    print(f"Saved results to {manually_de_duped_file}")
