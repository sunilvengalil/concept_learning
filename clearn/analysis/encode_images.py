import os
import pandas as pd
import numpy as np
from scipy.special import softmax

from clearn.utils.utils import get_latent_vector_column
from clearn.config.common_path import get_encoded_csv_file


def evaluate(model, train_val_data_iterator, epoch, dataset_type="train", save_results=True):
    encoded_df = None
    while train_val_data_iterator.has_next(dataset_type):
        batch_images, batch_labels, _ = train_val_data_iterator.get_next_batch(dataset_type)
        if batch_images.shape[0] < model.exp_config.BATCH_SIZE:
            train_val_data_iterator.reset_counter(dataset_type)
            break

        z, z, z, labels_predicted = model.encode(batch_images)
        labels_predicted = softmax(labels_predicted)
        labels_predicted = np.argmax(labels_predicted, axis=1)
        z_dim = z.shape[1]
        mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(z_dim)
        # TODO do this using numpy api
        labels = np.zeros(model.exp_config.BATCH_SIZE)
        i = 0
        for lbl in batch_labels:
            labels[i] = np.where(lbl == 1)[0][0]
            i += 1
        # print("labels_predicted shape",labels_predicted.shape)
        temp_df = pd.DataFrame(z, columns=z_col_names)
        temp_df["label"] = labels
        temp_df["label_predicted"] = labels_predicted
        if encoded_df is not None:
            encoded_df = pd.concat([encoded_df, temp_df])
        else:
            encoded_df = temp_df
    print(model.exp_config.ANALYSIS_PATH)

    if save_results:
        output_csv_file = get_encoded_csv_file(model.exp_config, epoch, dataset_type)
        encoded_df.to_csv(os.path.join(model.exp_config.ANALYSIS_PATH, output_csv_file), index=False)
    return encoded_df

def encode_images(model, train_val_data_iterator, exp_config, epoch, dataset_type="train", save_results=True):
    encoded_df = None
    while train_val_data_iterator.has_next(dataset_type):
        batch_images, batch_labels, _ = train_val_data_iterator.get_next_batch(dataset_type)
        if batch_images.shape[0] < exp_config.BATCH_SIZE:
            train_val_data_iterator.reset_counter(dataset_type)
            break

        mu, sigma, z, labels_predicted = model.encode(batch_images)
        labels_predicted = softmax(labels_predicted)
        labels_predicted = np.argmax(labels_predicted, axis=1)
        z_dim = z.shape[1]
        mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(z_dim)
        # TODO do this using numpy api
        labels = np.zeros(exp_config.BATCH_SIZE)
        i = 0
        for lbl in batch_labels:
            labels[i] = np.where(lbl == 1)[0][0]
            i += 1
        #print("labels_predicted shape",labels_predicted.shape)
        temp_df1 = pd.DataFrame(mu, columns=mean_col_names)
        temp_df2 = pd.DataFrame(sigma, columns=sigma_col_names)
        temp_df3 = pd.DataFrame(z, columns=z_col_names)
        temp_df = pd.concat([temp_df1, temp_df2, temp_df3], axis=1)
        temp_df["label"] = labels
        temp_df["label_predicted"] = labels_predicted
        if encoded_df is not None:
            encoded_df = pd.concat([encoded_df, temp_df])
        else:
            encoded_df = temp_df
    print(exp_config.ANALYSIS_PATH)

    if save_results:
        output_csv_file = get_encoded_csv_file(exp_config, epoch, dataset_type)
        encoded_df.to_csv(os.path.join(exp_config.ANALYSIS_PATH, output_csv_file), index=False)

    return encoded_df

from clearn.analysis.encode_decode import encode_and_get_features

def encode_images_and_get_features(model, train_val_data_iterator, exp_config, epoch, dataset_type="train", save_results=True):
    encoded_df = None
    while train_val_data_iterator.has_next(dataset_type):
        batch_images, batch_labels,_ = train_val_data_iterator.get_next_batch(dataset_type)
        if batch_images.shape[0] < exp_config.BATCH_SIZE:
            train_val_data_iterator.reset_counter(dataset_type)
            break

        #mu, sigma, z = model.encode(batch_images)
        mu, sigma, z, dense2_ens, reshapeds, conv2_ens, conv1_ens = encode_and_get_features(model,
                                                                                            batch_images,
                                                                                            exp_config.BATCH_SIZE,
                                                                                            exp_config.Z_DIM)
        z_dim = z.shape[1]
        mean_col_names, sigma_col_names, z_col_names,l3_col_names = get_latent_vector_column(z_dim)

        # TODO do this using numpy api
        labels = np.zeros(exp_config.BATCH_SIZE)
        i = 0
        for lbl in batch_labels:
            labels[i] = np.where(lbl == 1)[0][0]
            i += 1

        temp_df1 = pd.DataFrame(mu, columns=mean_col_names)
        temp_df2 = pd.DataFrame(sigma, columns=sigma_col_names)
        temp_df3 = pd.DataFrame(z, columns=z_col_names)
        temp_df4 = pd.DataFrame(dense2_ens, columns=l3_col_names)

        temp_df = pd.concat([temp_df1, temp_df2, temp_df3, temp_df4], axis=1)
        temp_df["label"] = labels
        if encoded_df is not None:
            encoded_df = pd.concat([encoded_df, temp_df])
        else:
            encoded_df = temp_df
    if save_results:
        output_csv_file = get_encoded_csv_file(exp_config, epoch, dataset_type)
        file_path = os.path.join(exp_config.ANALYSIS_PATH, output_csv_file)
        print(f"Saving features in {file_path} ")

        encoded_df.to_csv(file_path, index=False)
    return encoded_df
