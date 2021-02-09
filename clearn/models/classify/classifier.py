# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.special import softmax

from clearn.config import ExperimentConfig
from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.models.model import Model
from clearn.utils import prior_factory as prior

import tensorflow as tf
from clearn.utils.tensorflow_wrappers import linear
from clearn.utils.utils import get_latent_vector_column


class ClassifierModel(Model):
    _model_name = "ClassifierModel"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 dao=IDao,
                 num_units_in_layer=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 test_data_iterator=None
                 ):
        super().__init__(exp_config, sess, epoch, dao=dao, test_data_iterator=test_data_iterator)

        self.metrics_to_compute = ["accuracy"]
        self.metrics = dict()
        self.metrics[ClassifierModel.dataset_type_train] = dict()
        self.metrics[ClassifierModel.dataset_type_test] = dict()
        self.metrics[ClassifierModel.dataset_type_val] = dict()

        for metric in self.metrics_to_compute:
            self.metrics[ClassifierModel.dataset_type_train][metric] = []
            self.metrics[ClassifierModel.dataset_type_val][metric] = []
            self.metrics[ClassifierModel.dataset_type_test][metric] = []


        # test
        self.sample_num = 64  # number of generated images to be saved
        self.num_images_per_row = 4  # should be a factor of sample_num
        self.label_dim = self.dao.num_classes  # one hot encoding for 10 classes

        if num_units_in_layer is None or len(num_units_in_layer) == 0:
            self.n = [64, 128, 32, exp_config.Z_DIM * 2]
        else:
            self.n = num_units_in_layer

        self.mu = tf.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM], name='mu')
        self.sigma = tf.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM], name='sigma')
        self.images = None
        # self._set_model_parameters()
        self._build_model()
        # initialize all variables
        tf.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)
        self.counter, self.start_batch_id, self.start_epoch = self._initialize(read_from_existing_checkpoint,
                                                                               check_point_epochs)

    def _encoder(self, x, reuse=False):
        pass

    def _build_model(self):
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
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

    def get_trainable_vars(self):
        return tf.trainable_variables()

    def train(self, train_val_data_iterator):
        start_batch_id = self.start_batch_id
        start_epoch = self.start_epoch
        self.num_batches_train = train_val_data_iterator.get_num_samples("train") // self.exp_config.BATCH_SIZE

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for batch in range(start_batch_id, self.num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images, _, manual_labels = train_val_data_iterator.get_next_batch("train")
                if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                    break

                _, summary_str, loss, supervised_loss = self.sess.run([self.optim,
                                                                       self.merged_summary_op,
                                                                       self.loss,
                                                                       self.supervised_loss],
                                                                      feed_dict={self.inputs: batch_images,
                                                                                 self.labels: manual_labels[:,
                                                                                              :self.dao.num_classes]})

                self.counter += 1
                self.num_steps_completed = batch + 1
                self.writer.add_summary(summary_str, self.counter - 1)

            self.num_training_epochs_completed = epoch + 1
            print(f"Completed {epoch} epochs")
            if self.exp_config.run_evaluation_during_training:
                if np.mod(epoch, self.exp_config.eval_interval_in_epochs) == 0:
                    train_val_data_iterator.reset_counter("train")
                    train_val_data_iterator.reset_counter("val")
                    self.evaluate(train_val_data_iterator, "train")
                    self.evaluate(train_val_data_iterator, "val")
                    if self.test_data_iterator is not None:
                        self.evaluate(self.test_data_iterator, dataset_type="test")
                        self.test_data_iterator.reset_counter("test")

                    for metric in self.metrics_to_compute:
                        print(f"Accuracy: train: {self.metrics[ClassifierModel.dataset_type_train][metric][-1]}")
                        print(f"Accuracy: val: {self.metrics[ClassifierModel.dataset_type_val][metric][-1]}")
                        print(f"Accuracy: test: {self.metrics[ClassifierModel.dataset_type_test][metric][-1]}")

            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            train_val_data_iterator.reset_counter("train")
            if np.mod(epoch, self.exp_config.model_save_interval) == 0:
                print("Saving check point", self.exp_config.TRAINED_MODELS_PATH)
                self.save(self.exp_config.TRAINED_MODELS_PATH, self.counter)

            # save metrics
            if "accuracy" in self.metrics_to_compute:
                df = pd.DataFrame(self.metrics["train"]["accuracy"], columns=["epoch", "train_accuracy"])
                df["val_accuracy"] = np.asarray(self.metrics["val"]["accuracy"])[:, 1]
                df["test_accuracy"] = np.asarray(self.metrics["test"]["accuracy"])[:, 1]
                df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, f"accuracy_{start_epoch}.csv"),
                          index=False)
                max_accuracy = df["test_accuracy"].max()
                print("Max test accuracy", max_accuracy)

    def evaluate(self, data_iterator, dataset_type="train", num_batches_train=0, save_images=True):
        if num_batches_train == 0:
            num_batches_train = self.num_batches_train
        print(
            f"Running evaluation after epoch:{self.num_training_epochs_completed} and step:{self.num_steps_completed} ")

        labels_predicted = None
        labels = None
        z = None
        batch_no = 1
        while data_iterator.has_next(dataset_type):
            batch_images, batch_labels, _ = data_iterator.get_next_batch(dataset_type)
            # skip last batch
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break
            z_for_batch, y_pred = self.encode(batch_images)
            labels_predicted_for_batch = np.argmax(softmax(y_pred), axis=1)
            labels_for_batch = np.argmax(batch_labels, axis=1)
            if labels_predicted is None:
                labels_predicted = labels_predicted_for_batch
                labels = labels_for_batch
            else:
                labels_predicted = np.hstack([labels_predicted, labels_predicted_for_batch])
                labels = np.hstack([labels, labels_for_batch])
            if self.exp_config.return_latent_vector:
                if z is None:
                    z = z_for_batch
                else:
                    z = np.hstack([z, z_for_batch])
            batch_no += 1

        if "accuracy" in self.metrics_to_compute:
            accuracy = accuracy_score(labels, labels_predicted)
            self.metrics[dataset_type]["accuracy"].append([self.num_training_epochs_completed, accuracy])

        encoded_df = pd.DataFrame(np.transpose(np.vstack([labels, labels_predicted])),
                                  columns=["label", "label_predicted"])
        if self.exp_config.return_latent_vector:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(self.exp_config.Z_DIM)
            encoded_df[z_col_names] = z
        if self.exp_config.write_predictions:
            output_csv_file = get_encoded_csv_file(self.exp_config,
                                                   self.num_training_epochs_completed,
                                                   dataset_type
                                                   )
            print("Saving evaluation results to ", self.exp_config.ANALYSIS_PATH)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)
        return encoded_df

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self._model_name_, self.exp_config.dataset_name,
            self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

    def encode(self, images):
        z, y_pred = self.sess.run([self.z, self.y_pred],
                                  feed_dict={self.inputs: images})
        return z, y_pred

    def classify(self, images):
        logits = self.sess.run([self.y_pred],
                               feed_dict={self.inputs: images})

        return logits
