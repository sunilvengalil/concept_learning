import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os


def read_label(label_file, num_label_files):
    labels = {}
    for file_number in range(num_label_files):
        label_df = pd.read_csv(label_file.format(file_number))
        labels[file_number] = label_df["label"].values
    return labels


def get_label_reconstructed(_df, num_rows_per_image, num_digits_per_row):
    labels = np.ones(num_rows_per_image * num_digits_per_row) * -2
    _df = _df.fillna("xxxx")
    for row in _df.iterrows():
        text_ = [-1] * num_digits_per_row

        row_text_ = row[1]["text"]
        if isinstance(row_text_, float):
            row_text_ = str(row_text_)
        row_text_ = row_text_.strip()
        if len(row_text_) != 0:
            if len(row_text_) < 4:
                for i in range(4-len(row_text_)):
                    text_[i] = 0
            offset = 4 - len(row_text_)
            for i, c in enumerate(row_text_):
                if c.isdigit():
                    text_[i+offset] = int(c)
                elif c == 'x':
                    text_[i+offset] = -1
                else:
                    raise Exception("Invalid character in annotated data - ", row[1]["num_rows_annotated"])

        for i in range(num_digits_per_row)  :
            offset = (row[1]["num_rows_annotated"] - 1) * num_digits_per_row
            labels[i + offset] = text_[i]
    return labels


def get_annotations(annotated_path):
    df = None
    for annotation_file in os.listdir(annotated_path):
        if annotation_file.rsplit(".", 1)[1] == "csv":
            annotation_csv = os.path.join(annotated_path, annotation_file)
            _df = pd.read_csv(annotation_csv)
            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df])
    unique = df.groupby(["epoch", "step"]).size().reset_index().rename(columns={0: 'count'})
    return df, unique


def _compute_accuracy(df,
                      step,
                      epoch,
                      num_label_files,
                      labels,
                      num_rows_per_image,
                      num_digits_per_row
                      ):
    df1 = df[(df["epoch"] == epoch) & (df["step"] == step)]
    labels_batch = []
    reconstructed_batch = []
    for image_no in range(num_label_files):
        _df = df1[df1["_idx"] == image_no]
        if _df.shape[0] > 0 :
            try:
                reconstructed = get_label_reconstructed(_df[["num_rows_annotated", "text"]],
                                                        num_rows_per_image,
                                                        num_digits_per_row)
            except Exception as e:
                print("Invalid character in annotation,epoch {:01d} , step {:01d}".format(epoch,step))
                print(str(e))
                continue
            _reconstructed_indices = reconstructed != -2
            reconstructed_batch.extend(reconstructed[_reconstructed_indices])
            labels_batch.extend(labels[image_no][_reconstructed_indices])
    accuracy = accuracy_score(labels_batch,reconstructed_batch)
    return accuracy


def compute_accuracy(labels, gt_dir,
                     max_epoch,
                     num_label_files,
                     num_rows_per_image,
                     num_digits_per_row,
                     eval_interval
                     ):
    df, unique = get_annotations(gt_dir)
    unique = unique[unique["count"] > 10]

    accuracies = []
    total_batches_finished = []
    for unique_combination in unique.iterrows():
        epoch = unique_combination[1]["epoch"]
        step = unique_combination[1]["step"]
        accuracy = _compute_accuracy(df,
                                     step,
                                     epoch,
                                     num_label_files,
                                     labels,
                                     num_rows_per_image,
                                     num_digits_per_row,
                                     )
        accuracies.append(accuracy)
        total_batches_finished.append(epoch * 935 + step * eval_interval)
    accuracy_df = pd.DataFrame(
        {"Total_Batches_Finished": total_batches_finished, "Epochs": np.asarray(total_batches_finished) / 935,
         "Accuracy": accuracies})
    accuracy_df = accuracy_df[accuracy_df["Epochs"] < max_epoch]
    return accuracy_df
