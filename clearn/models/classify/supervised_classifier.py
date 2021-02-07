# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score

from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao

from clearn.models.classify.classifier import ClassifierModel
import tensorflow as tf

from clearn.utils.data_loader import DataIterator
from clearn.utils.tensorflow_wrappers import conv2d, linear, lrelu
from clearn.utils.utils import get_latent_vector_column


class SupervisedClassifierModel(ClassifierModel):
    _model_name_ = "SupervisedClassifierModel"
    dataset_type_test = "test"
    dataset_type_train = "train"
    dataset_type_val = "val"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 num_units_in_layer=None,
                 check_point_epochs=None,
                 dao: IDao = MnistDao(),
                 test_data_iterator: DataIterator = None,
                 ):
        super().__init__(exp_config, sess, epoch, check_point_epochs=check_point_epochs, dao=dao)
        self.test_data_iterator = test_data_iterator
        self.label_dim = dao.num_classes
        if num_units_in_layer is None or len(num_units_in_layer) == 0:
            self.n = [128, 64, 32, exp_config.Z_DIM]
        else:
            self.n = num_units_in_layer
        self.sample_num = 64  # number of generated images to be saved
        self.num_images_per_row = 4  # should be a factor of sample_num
        self.images = None
        # graph inputs for visualize training results
        if epoch != -1:
            self.epoch = epoch

        self.metrics_to_compute = ["accuracy"]
        self.metrics = dict()
        self.metrics[SupervisedClassifierModel.dataset_type_train] = dict()
        self.metrics[SupervisedClassifierModel.dataset_type_test] = dict()
        self.metrics[SupervisedClassifierModel.dataset_type_val] = dict()

        for metric in self.metrics_to_compute:
            self.metrics[SupervisedClassifierModel.dataset_type_train][metric] = []
            self.metrics[SupervisedClassifierModel.dataset_type_val][metric] = []
            self.metrics[SupervisedClassifierModel.dataset_type_test][metric] = []

    # def _set_model_parameters(self):
    #     self.strides = [2, 2]

    def _encoder(self, x, reuse=False):
        # Encoder models the probability  P(z/X)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        w = dict()
        b = dict()
        with tf.compat.v1.variable_scope("encoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                layer_num = 0
                self.conv1 = lrelu(
                    conv2d(x, self.n[layer_num], 3, 3, self.strides[0], self.strides[0], name='en_conv1'))
                layer_num += 1
                final_conv = self.conv1
                if self.exp_config.num_decoder_layer == 4:
                    self.conv2 = lrelu((conv2d(self.conv1,
                                               self.n[layer_num],
                                               3, 3, self.strides[1], self.strides[1], name='en_conv2')))
                    layer_num += 1
                    final_conv = self.conv2
                self.reshaped_en = tf.reshape(final_conv, [self.exp_config.BATCH_SIZE, -1])
                # self.dense2_en = lrelu(linear(self.reshaped_en, self.n[layer_num], scope='en_fc3'))
                layer_num += 1
            elif self.exp_config.activation_hidden_layer == "LINEAR":
                self.conv1 = conv2d(x, self.n[0], 3, 3, self.strides[0], self.strides[0], name='en_conv1')
                self.conv2 = (conv2d(self.conv1, self.n[1], 3, 3, self.strides[1], self.strides[1], name='en_conv2'))
                self.reshaped_en = tf.reshape(self.conv2, [self.exp_config.BATCH_SIZE, -1])
                self.dense2_en = linear(self.reshaped_en, self.n[2], scope='en_fc3')
            else:
                raise Exception(f"Activation {self.exp_config.activation_hidden_layer} not supported")

            z, w["en_fc4"], b["en_fc4"] = linear(self.reshaped_en, self.exp_config.Z_DIM,
                                                 scope='en_fc4',
                                                 with_w=True)

        return z

    def _build_model(self):
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
        self.strides = [2, 2]

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.labels = tf.compat.v1.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        """ Loss Function """
        # encoding
        self.z = self._encoder(self.inputs, reuse=False)

        # supervised loss for labelled samples
        self.y_pred = linear(self.z, self.dao.num_classes)
        self.compute_and_optimize_loss()

    def compute_and_optimize_loss(self):
        self.supervised_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                                         logits=self.y_pred
                                                                         )
        self.loss = self.exp_config.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.compat.v1.trainable_variables()
        # TODO add beta1 parameter from exp_config
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optim = tf.compat.v1.train.AdamOptimizer(self.exp_config.learning_rate,
                                                          self.exp_config.beta1_adam).minimize(self.loss,
                                                                                               var_list=t_vars)

        """ Summary """
        tf.compat.v1.summary.scalar("Supervised Loss", self.supervised_loss)
        tf.compat.v1.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.compat.v1.summary.merge_all()


    def train(self, train_val_data_iterator):
        counter = self.counter
        start_batch_id = self.start_batch_id
        start_epoch = self.start_epoch
        num_batches_train = train_val_data_iterator.get_num_samples("train") // self.exp_config.BATCH_SIZE

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images, _, manual_labels = train_val_data_iterator.get_next_batch("train")
                if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                    break

                # update autoencoder
                _, summary_str, loss, supervised_loss = self.sess.run([self.optim,
                                                                       self.merged_summary_op,
                                                                       self.loss,
                                                                       self.supervised_loss],
                                                                      feed_dict={self.inputs: batch_images,
                                                                                 self.labels: manual_labels[:,
                                                                                              :self.dao.num_classes]})

                counter += 1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            print(f"Completed {epoch} epochs")
            if self.exp_config.run_evaluation_during_training:
                if np.mod(epoch, self.exp_config.eval_interval_in_epochs) == 0:
                    train_val_data_iterator.reset_counter("train")
                    train_val_data_iterator.reset_counter("val")
                    self.evaluate(train_val_data_iterator, epoch, "train")
                    self.evaluate(train_val_data_iterator, epoch, "val")
                    if self.test_data_iterator is not None:
                        self.evaluate(self.test_data_iterator, epoch, dataset_type="test")
                        self.test_data_iterator.reset_counter("test")
            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            for metric in self.metrics_to_compute:
                print(f"Accuracy: train: {self.metrics[SupervisedClassifierModel.dataset_type_train][metric][-1]}")
                print(f"Accuracy: val: {self.metrics[SupervisedClassifierModel.dataset_type_val][metric][-1]}")
                print(f"Accuracy: test: {self.metrics[SupervisedClassifierModel.dataset_type_test][metric][-1]}")

            start_batch_id = 0
            # save model
            if np.mod(epoch, self.exp_config.model_save_interval) == 0:
                self.save(self.exp_config.TRAINED_MODELS_PATH, counter)

            # save metrics
            # TODO save all metrics. not just accuracy
            if "accuracy" in self.metrics_to_compute:
                df = pd.DataFrame(self.metrics["train"]["accuracy"], columns=["epoch", "train_accuracy"])
                df["val_accuracy"] = np.asarray(self.metrics["val"]["accuracy"])[:, 1]
                df["test_accuracy"] = np.asarray(self.metrics["test"]["accuracy"])[:, 1]
                df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, f"accuracy_{start_epoch}.csv"),
                          index=False)
                max_accuracy = df["test_accuracy"].max()
                print(max_accuracy)

        # save model for final step
        self.save(self.exp_config.TRAINED_MODELS_PATH, counter)

    def evaluate(self, train_val_data_iterator, epoch=-1, dataset_type="train", return_latent_vector=False):
        if epoch == -1:
            epoch = self.start_epoch
        labels_predicted = None
        labels = None
        mu = None
        sigma = None
        z = None
        batch_no = 1
        while train_val_data_iterator.has_next(dataset_type):
            batch_images, batch_labels, _ = train_val_data_iterator.get_next_batch(dataset_type)
            # skip last batch
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                train_val_data_iterator.reset_counter(dataset_type)
                break
            mu_for_batch, sigma_for_batch, z_for_batch, y_pred = self.encode(batch_images)
            labels_predicted_for_batch = np.argmax(softmax(y_pred), axis=1)
            labels_for_batch = np.argmax(batch_labels, axis=1)
            if labels_predicted is None:
                labels_predicted = labels_predicted_for_batch
                labels = labels_for_batch
            else:
                labels_predicted = np.hstack([labels_predicted, labels_predicted_for_batch])
                labels = np.hstack([labels, labels_for_batch])
            if return_latent_vector:
                if mu is None:
                    mu = mu_for_batch
                    sigma = sigma_for_batch
                    z = z_for_batch
                else:
                    mu = np.hstack([mu, mu_for_batch])
                    sigma = np.hstack([sigma, sigma_for_batch])
                    z = np.hstack([z, z_for_batch])
            batch_no += 1

        if "accuracy" in self.metrics_to_compute:
            accuracy = accuracy_score(labels, labels_predicted)
            self.metrics[dataset_type]["accuracy"].append([epoch, accuracy])

        encoded_df = pd.DataFrame(np.transpose(np.vstack([labels, labels_predicted])),
                                  columns=["label", "label_predicted"])
        if return_latent_vector:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(self.exp_config.Z_DIM)
            encoded_df[mean_col_names] = mu
            encoded_df[sigma_col_names] = sigma
            encoded_df[z_col_names] = z
        if self.exp_config.write_predictions:
            output_csv_file = get_encoded_csv_file(self.exp_config, epoch, dataset_type)
            print("Saving evaluation results to ", self.exp_config.ANALYSIS_PATH)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)
        return encoded_df
