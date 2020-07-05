import pandas as pd
import numpy as np


def read_label(label_file, num_label_files):
    labels = {}
    for file_number in range(num_label_files):
        label_df = pd.read_csv(label_file.format(file_number))
        #label_df = pd.read_csv(label_file)
        labels[file_number] = label_df["label"].values
    return labels


def get_label_reconstructed(_df, num_rows_per_image, num_digits_per_row):
    labels =  np.ones(num_rows_per_image * num_digits_per_row) * -2
    _df = _df.fillna("xxxx")
    for row in _df.iterrows():
        text_ = [-1] * num_digits_per_row

        row_text_ = row[1]["text"]
        if isinstance(row_text_,float):
            row_text_ = str(row_text_)
        row_text_ = row_text_.strip()
        if len(row_text_) != 0:
            if len(row_text_) < 4:
                for i in range(4-len(row_text_)):
                    text_[i] = 0
            offset = 4 - len(row_text_)
            for i,c in enumerate(row_text_):
                if c.isdigit():
                    text_[i+offset] = int(c)
                elif c == 'x':
                    text_[i+offset] = -1
                else:
                    raise Exception("Invalid character in annotated data - ",row[1]["num_rows_annotated"])

        for i in range(num_digits_per_row)  :
            offset = (row[1]["num_rows_annotated"] - 1) * num_digits_per_row
            labels[i + offset] = text_[i]

    return labels