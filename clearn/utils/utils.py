"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division

from typing import List, Tuple

import pandas as pd
import scipy.misc
from skimage import img_as_ubyte

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import os
import imageio
from collections import defaultdict

from clearn.config import ExperimentConfig
from clearn.utils.data_loader import load_test_raw_data

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

SIGNIFICANT_THRESHOLD = 0.15


def is_convolutional_layer(layer_num, num_units, num_dense_layers):
    if layer_num >= len(num_units) - num_dense_layers :
        return False
    else:
        return True


def get_padding_info(exp_config:ExperimentConfig,
                     image_shape: Tuple[int]
                     ):
    num_units = exp_config.num_units
    num_dense_layers = exp_config.num_dense_layers
    strides = exp_config.strides
    image_sizes = []
    padding_added_row = []
    padding_added_col = []
    print("Image shape", image_shape)
    height = image_shape[0]
    width = image_shape[1]
    channels =  image_shape[2]
    image_sizes.append((height, width, channels))
    layer_num = 0
    for layer_num, stride in enumerate(strides[:-1]):
        if not is_convolutional_layer(layer_num, num_units, num_dense_layers):
            image_sizes.append(num_units[layer_num])
            continue
        height, width = get_image_size(height, padding_added_col, padding_added_row, stride, width)
        image_sizes.append((height, width, num_units[layer_num]))
    if is_convolutional_layer(layer_num + 1, num_units, num_dense_layers):
        height, width = get_image_size(height, padding_added_col, padding_added_row, strides[-1], width)
        image_sizes.append((height, width, num_units[layer_num + 1]))
    else:
        image_sizes.append(exp_config.Z_DIM)
    print(image_sizes)
    print(num_units)
    return padding_added_row, padding_added_col, image_sizes


def get_image_size(height, padding_added_col, padding_added_row, stride, width):
    if height % stride != 0:
        num_pads = stride - height % stride
        pad_left = num_pads // 2
        pad_right = num_pads - pad_left
        padding_added_row.append((pad_left, pad_right))
        height = (height // stride) + 1
    else:
        padding_added_row.append((0, 0))
        height = height // stride
    if width % stride != 0:
        num_pads = stride - width % stride
        pad_left = num_pads // 2
        pad_right = num_pads - pad_left
        padding_added_col.append((pad_left, pad_right))
        width = (width // stride) + 1
    else:
        padding_added_col.append((0, 0))
        width = width // stride
    return height, width


def check_and_create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_significant_dimensions(lv_dimensions, file_name):
    """
    Find the indices with most significant values on the vector `lv_dimensions`
    @param lv_dimensions vector for which most significant value needs to be found
    @param file_name full path and name of the file for saving the elbow graph
    @returns a 1-d ndarray with values as indices of most significant values in input vector
    """
    plt.figure()
    kmeans_model = KMeans()
    visualizer = KElbowVisualizer(kmeans_model, k=(1, 10))
    visualizer.fit(lv_dimensions)
    visualizer.show(file_name)
    kmeans_model.n_clusters = visualizer.elbow_value_
    cluster_labels = kmeans_model.fit_predict(lv_dimensions)
    cluster_centers = kmeans_model.cluster_centers_
    cluster_centers_indices_sorted = np.squeeze(cluster_centers.argsort(axis=0))
    print(cluster_centers[cluster_centers_indices_sorted[0]],
          cluster_centers[cluster_centers_indices_sorted[-1]],
          cluster_centers[cluster_centers_indices_sorted[0]] / cluster_centers[cluster_centers_indices_sorted[-1]]
          )
    norm_of_sum_of_vectors_in_cluster = np.zeros_like(cluster_centers)
    for i in range(len(cluster_centers)):
        norm_of_sum_of_vectors_in_cluster[i] = np.linalg.norm(lv_dimensions[np.where(cluster_labels == i)[0]])
    max_norm = np.max(norm_of_sum_of_vectors_in_cluster)
    norm_of_sum_of_vectors_in_cluster = norm_of_sum_of_vectors_in_cluster / max_norm
    dims_to_add = np.where(norm_of_sum_of_vectors_in_cluster > SIGNIFICANT_THRESHOLD)[0]

    significant_dimensions = []
    for i in range(len(lv_dimensions)):
        if cluster_labels[i] in dims_to_add:
            significant_dimensions.append(i)

    return np.asarray(significant_dimensions), cluster_centers


def find_absolute_largest_weights(weights, num_out_units):
    """
    Find the maximum of absolute value of weights (MAVoW) connecting to each output unit
    @return returns an array of shape (num_out_units) containing MAVoW for each output unit
    """
    max_abs_for_each_out_dimension_weight = np.asarray([np.max(np.abs(weights[:, i])) for i in range(num_out_units)] )
    max_abs_for_each_out_dimension_weight = np.reshape(max_abs_for_each_out_dimension_weight, (num_out_units, 1))
    return max_abs_for_each_out_dimension_weight


def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    print("TODO  Fix the model analyzer issue to see model structure")
#    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_single_image(images, path, epoch, step, training_batch, eval_batch, eval_batch_size):
    """
    Save all the reconstructed images in a particular validation batch
    @:param images the reconstructed images in current batch
    @:param path path for saving the image
    @:param epoch number of training epochs completed
    @:param step number of steps within the epoch
    @:param eval_batch validation batch in which the image belongs to
    """
    for i in range(images.shape[0]):
        file = f"im_epoch_{epoch}_step_{step}_batch_{training_batch}_eval_image_id_{eval_batch * eval_batch_size + i }.png"
        image = inverse_transform(images[i])
        image = np.squeeze(image)
        imageio.imwrite(path + file, img_as_ubyte(image))


def save_image(image, size, image_file_name):
    return imsave(inverse_transform(np.asarray(image)), size, image_file_name)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    imageio.imwrite(path, img_as_ubyte(image))


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.

def inverse_transform(images):
    return (images + 1)/2.


def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    """ Drawing Tools """
    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb

    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def segregate_images_by_label(train_val_iterator, dataset_type="val"):
    labels = set()
    feature_shape = list(train_val_iterator.get_feature_shape())
    feature_shape.insert(0, 10)
    images_by_label = defaultdict(list)
    num_batches = 0
    while train_val_iterator.has_next(dataset_type):
        num_batches += 1
        images, label = train_val_iterator.get_next_batch(dataset_type)
        print(images.shape)
        for im, lbl in zip(images, label):
            _lbl = np.where(lbl == 1)[0][0]
            labels.add(_lbl)
            images_by_label[_lbl].append(im)
    return images_by_label


def get_latent_vector_column(z_dim, returnpredicted_proba_col_name=False):
    mean_col_names = ["mu_{}".format(i) for i in range(z_dim)]
    sigma_col_names = ["sigma_{}".format(i) for i in range(z_dim)]
    z_col_names = ["z_{}".format(i) for i in range(z_dim)]
    # TODO fix this
    l3_col_names = ["l3_{}".format(i) for i in range(32)]
    if returnpredicted_proba_col_name:
        predicted_proba_col_names = ["predicted_proba_{}".format(i) for i in range(z_dim)]
        return mean_col_names, sigma_col_names, z_col_names, l3_col_names, predicted_proba_col_names
    else:
        return mean_col_names, sigma_col_names, z_col_names, l3_col_names


def get_mean(i, df, mean_col_names):
    df_0 = df[df["label"] == i]
    mu_0 = df_0[mean_col_names].values
    mu_0 = mu_0.mean(axis=0)
    return mu_0


def get_min(i, df, mean_col_names):
    df_0 = df[df["label"] == i]
    mu_0 = df_0[mean_col_names].values
    mu_0 = mu_0.min(axis=0)
    return mu_0


def get_max(i,df,mean_col_names):
    df_0 = df[df["label"] == i]
    mu_0 = df_0[mean_col_names].values
    mu_0 = mu_0.max(axis=0)
    return mu_0


def get_labels(data_set_path):
    x, y = load_test_raw_data(data_set_path)
    labels = np.unique(y)
    return labels


def get_latent_vector(dataset_path, z_dim, df, labels=None):
    if labels is None:
        labels = get_labels(dataset_path)
    _, _, z_col_names,_ = get_latent_vector_column(z_dim)
    mu_mean = []
    for i in labels:
        mu_mean.append( get_mean(i, df, z_col_names))
    return labels, mu_mean


"""
Computes the histogram or probability mass function of y given z
@:param data_df DataFrame containing the samples
@:param y_column_name Name of the column which have categorical y values
@:param z_dim dimension of z
@:param normalized Returns probability mass function if True. Otherwise returns un-normalized probability
Note:- It is assumed that  data_df has columns with names as returned by get_latent_vector_column().
It should also have a categorical column with name specified by parameter y_column_name
"""


def get_pmf_y_given_z(data_df, y_column_name, z_dim, normalized=True):
    _,  _, z_col_names,_ = get_latent_vector_column(z_dim)
    un_normalized = data_df[[z_col_names[0],
                             y_column_name
                             ]].groupby(by=y_column_name).count()[z_col_names[0]]
    pmf_y_given_z = un_normalized / data_df.shape[0]
    if normalized:
        return pd.Series(pmf_y_given_z, index=un_normalized.index.values)
    else:
        return un_normalized


# TODO fix this to remove the parameter num_label_files. Instead find and read all files with the
# specific format
# Read labels from label file and returns a dictionary of in the format {file_num:label_data_frame}
# TODO move  this  method to train/iterator
def read_label(label_file_prefix, num_label_files):
    labels = {}
    for file_number in range(num_label_files):
        label_df = pd.read_csv(label_file_prefix.format(file_number))
        labels[file_number] = label_df["label"].values
    return labels


# Construct and return an ndarray of manually given labels
# TODO move this method to annotation utils
# annotated_df annotated df
# TODO remove hard coding of string
def get_label_reconstructed(annotated_df, num_rows_per_image, num_digits_per_row):
    # Initialize all labels with -2
    labels = np.ones(num_rows_per_image * num_digits_per_row) * -2
    annotated_df = annotated_df.fillna("xxxx")
    for row in annotated_df.iterrows():
        text = [-1] * num_digits_per_row
        row_text = row[1]["text"]
        if isinstance(row_text, float):
            row_text = str(row_text)
        row_text = row_text.strip()
        if len(row_text) != 0:
            if len(row_text) < num_digits_per_row:
                # TODO fix this. instead read each row as a string
                for i in range(num_digits_per_row - len(row_text)):
                    text[i] = 0
            offset = num_digits_per_row - len(row_text)
            for i, c in enumerate(row_text):
                if c.isdigit():
                    text[i + offset] = int(c)
                elif c == 'x':
                    text[i + offset] = -1
                else:
                    raise Exception("Invalid character in annotated data - ", row[1]["num_rows_annotated"])

        for i in range(num_digits_per_row):
            offset = (row[1]["num_rows_annotated"] - 1) * num_digits_per_row
            labels[i + offset] = text[i]

    return labels
