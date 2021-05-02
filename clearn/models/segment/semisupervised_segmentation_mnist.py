# -*- coding: utf-8 -*-
from __future__ import division

import json
import traceback
import os
from collections import defaultdict
from typing import List, DefaultDict

import numpy as np
import pandas as pd
from statistics import mean

from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.models.architectures.custom.tensorflow_graphs import  fcnn_n_layer, fully_deconv_n_layer
from clearn.models.classify.classifier import ClassifierModel
from clearn.models.vae import VAE
from clearn.utils import prior_factory as prior
from clearn.utils.retention_policy.policy import RetentionPolicy
from clearn.utils.utils import save_image, get_latent_vector_column
from clearn.utils.dir_utils import get_eval_result_dir
from scipy.special import softmax
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from clearn.utils.tensorflow_wrappers import linear


class SemiSupervisedSegmenterMnist(VAE):
    _model_name_ = "SemiSupervisedSegmenterMnist"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 dao: IDao,
                 test_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 num_concepts=20
                 ):
        # Whether the sample was manually annotated.
        self.is_manual_annotated = placeholder(tf.float32, [exp_config.BATCH_SIZE], name="is_manual_annotated")
        self.labels = placeholder(tf.float32, [exp_config.BATCH_SIZE, dao.num_classes], name='manual_label')
        self.num_concepts = num_concepts
        super().__init__(exp_config=exp_config,
                         sess=sess,
                         epoch=epoch,
                         dao=dao,
                         test_data_iterator=test_data_iterator,
                         read_from_existing_checkpoint=read_from_existing_checkpoint,
                         check_point_epochs=check_point_epochs,
                         )

        #self.metrics_to_compute = ["accuracy", "reconstruction_loss", "intersection_over_union", ]

        self.metrics_to_compute = ["reconstruction_loss" ]
        self.metrics = dict()
        self.metrics[SemiSupervisedSegmenterMnist.dataset_type_train] = dict()
        self.metrics[SemiSupervisedSegmenterMnist.dataset_type_test] = dict()
        self.metrics[SemiSupervisedSegmenterMnist.dataset_type_val] = dict()
        for metric in self.metrics_to_compute:
            self.metrics[SemiSupervisedSegmenterMnist.dataset_type_train][metric] = []
            self.metrics[SemiSupervisedSegmenterMnist.dataset_type_val][metric] = []
            self.metrics[SemiSupervisedSegmenterMnist.dataset_type_test][metric] = []

    def _encoder(self, x, reuse=False):
        if len(self.exp_config.num_units) in (2, 3):
            gaussian_params = fcnn_n_layer(self, x, 2, reuse)
        else:
            raise Exception("Invalid configuration. Length of num_units should be 2 or 3")

        # The mu parameter is unconstrained
        # mu = gaussian_params[:, :, :, 0]
        mu = gaussian_params[:, :49]

        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        #stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, :, :, 1])
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, 49:])
        print("mu.shape", mu.shape)
        print("std.shape", stddev.shape)
        return mu, stddev

    def _decoder(self, z, reuse=False):
        print("decoder shape of z", z.shape)
        if len(self.exp_config.num_units) in (2, 3):
            return fully_deconv_n_layer(self, z, reuse)
        else:
            raise Exception("Invalid configuration. Length of num_units should be 2 or 3")

    def get_decoder_weights_bias(self):
        name_w_1 = "decoder/de_fc1/Matrix:0"
        name_w_2 = "decoder/de_dc3/w:0"
        name_w_3 = "decoder/de_dc4/w:0"

        name_b_1 = "decoder/de_fc1/bias:0"
        name_b_2 = "decoder/de_dc3/biases:0"
        name_b_3 = "decoder/de_dc4/biases:0"

        layer_param_names = [name_w_1,
                             name_b_1,
                             name_w_2,
                             name_b_2,
                             name_w_3,
                             name_b_3,
                             ]

        default_graph = tf.get_default_graph()
        params = [default_graph.get_tensor_by_name(tn) for tn in layer_param_names]
        param_values = self.sess.run(params)
        return {tn: tv for tn, tv in zip(layer_param_names, param_values)}

    def get_encoder_weights_bias(self):
        name_w_1 = "encoder/en_conv1/w:0"
        if len(self.exp_config.num_units) > 2:
            name_w_2 = "encoder/en_conv2/w:0"
        name_w_3 = "encoder/en_fc2/Matrix:0"
        name_w_4 = "encoder/en_fc3/Matrix:0"

        name_b_1 = "encoder/en_conv1/biases:0"
        if len(self.exp_config.num_units) > 2:
            name_b_2 = "encoder/en_conv2/biases:0"
        name_b_3 = "encoder/en_fc2/bias:0"
        name_b_4 = "encoder/en_fc3/bias:0"

        layer_param_names = [name_w_1,
                             name_b_1]
        if len(self.exp_config.num_units) > 2:
            layer_param_names.append(name_w_2)
            layer_param_names.append(name_b_2)
        layer_param_names.extend([ name_w_3,
                                   name_b_3,
                                   name_w_4,
                                   name_b_4
                                   ]
                                 )
        default_graph = tf.get_default_graph()
        params = [default_graph.get_tensor_by_name(tn) for tn in layer_param_names]
        param_values = self.sess.run(params)
        return {tn: tv for tn, tv in zip(layer_param_names, param_values)}

    def compute_and_optimize_loss(self):
        self.y_pred = linear(self.z, self.num_concepts)
        # self.supervised_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels,
        #                                                                  logits=self.y_pred,
        #                                                                  weights=self.is_manual_annotated
        #                                                                  )

        # self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood + \
        #             self.exp_config.beta * self.KL_divergence + \
        #             self.exp_config.supervise_weight * self.supervised_loss

        self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood + \
                    self.exp_config.beta * self.KL_divergence

        # self.loss = -evidence_lower_bound + self.exp_config.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.compat.v1.trainable_variables()
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optim = tf.compat.v1.train.AdamOptimizer(self.exp_config.learning_rate,
                                                          beta1=self.exp_config.beta1_adam) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        # self.fake_images = self._decoder(self.standard_normal, reuse=True)

        """ Summary """
        tf.compat.v1.summary.scalar("Negative Log Likelihood", self.neg_loglikelihood)
        tf.compat.v1.summary.scalar("K L Divergence", self.KL_divergence)
        #tf.compat.v1.summary.scalar("Supervised Loss", self.supervised_loss)

        tf.compat.v1.summary.scalar("Total Loss", self.loss)
        # final summary operations
        self.merged_summary_op = tf.compat.v1.summary.merge_all()

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
                batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

                # # update aut encoder and classifier parameters
                # _, summary_str, loss, nll_loss, nll_batch, kl_loss, supervised_loss = self.sess.run([self.optim,
                #                                                                                      self.merged_summary_op,
                #                                                                                      self.loss,
                #                                                                                      self.neg_loglikelihood,
                #                                                                                      self.marginal_likelihood,
                #                                                                                      self.KL_divergence,
                #                                                                                      self.supervised_loss],
                #                                                                                     feed_dict={
                #                                                                                         self.inputs: batch_images,
                #                                                                                         self.labels: manual_labels[
                #                                                                                                      :,
                #                                                                                                      :self.dao.num_classes],
                #                                                                                         self.is_manual_annotated: manual_labels[
                #                                                                                                                   :,
                #                                                                                                                   self.dao.num_classes],
                #                                                                                         self.standard_normal: batch_z}
                #                                                                                     )
                _, summary_str, loss, nll_loss, nll_batch, kl_loss = self.sess.run([self.optim,
                                                                                    self.merged_summary_op,
                                                                                    self.loss,
                                                                                    self.neg_loglikelihood,
                                                                                    self.marginal_likelihood,
                                                                                    self.KL_divergence
                                                                                    ],
                                                                                   feed_dict={self.inputs: batch_images,
                                                                                              self.labels: manual_labels[:, :self.dao.num_classes],
                                                                                              self.is_manual_annotated: manual_labels[:, self.dao.num_classes],
                                                                                              self.standard_normal: batch_z
                                                                                              }
                                                                                   )
                # print(f"Epoch: {epoch}/{batch}, Nll_loss shape: {nll_loss.shape}, Nll_batch: {nll_batch.shape}")
                #print(f"Epoch: {epoch}/{batch}, Nll_loss : {nll_loss} KLD:{kl_loss}  Supervised loss:{supervised_loss}")

                self.counter += 1
                self.num_steps_completed = batch + 1
                # self.writer.add_summary(summary_str, self.counter - 1)
            print(f"Epoch: {epoch}/{batch}, Nll_loss : {nll_loss}")
            self.num_training_epochs_completed = epoch + 1
            print(f"Completed {epoch} epochs")
            if self.exp_config.run_evaluation_during_training:
                print("Eval_interval", self.exp_config.eval_interval_in_epochs)
                if np.mod(epoch, self.exp_config.eval_interval_in_epochs) == 0:
                    train_val_data_iterator.reset_counter("val")
                    train_val_data_iterator.reset_counter("train")
                    self.evaluate(data_iterator=train_val_data_iterator,
                                  dataset_type="val")
                    self.evaluate(data_iterator=train_val_data_iterator,
                                  dataset_type="train")
                    if self.test_data_iterator is not None:
                        self.test_data_iterator.reset_counter("test")
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

        train_val_data_iterator.reset_counter("val")
        train_val_data_iterator.reset_counter("train")
        self.evaluate(data_iterator=train_val_data_iterator,
                      dataset_type="val")
        self.evaluate(data_iterator=train_val_data_iterator,
                      dataset_type="train")
        if self.test_data_iterator is not None:
            self.test_data_iterator.reset_counter("test")
            self.evaluate(self.test_data_iterator, dataset_type="test")
            self.test_data_iterator.reset_counter("test")

        for metric in self.metrics_to_compute:
            print(f"Accuracy: train: {self.metrics[ClassifierModel.dataset_type_train][metric][-1]}")
            print(f"Accuracy: val: {self.metrics[ClassifierModel.dataset_type_val][metric][-1]}")
            print(f"Accuracy: test: {self.metrics[ClassifierModel.dataset_type_test][metric][-1]}")

        # save metrics
        df = None
        for i, metric in enumerate(self.metrics_to_compute):
            column_name = f"train_{metric}"
            if i == 0:
                df = pd.DataFrame(self.metrics["train"][metric], columns=["epoch", column_name])
            else:
                df[column_name] = np.asarray(self.metrics["train"][metric])[:, 1]
            df[f"val_{metric}"] = np.asarray(self.metrics["val"][metric])[:, 1]
            df[f"test_{metric}"] = np.asarray(self.metrics["test"][metric])[:, 1]
            max_value = df[f"test_{metric}"].max()
            print(f"Max test {metric}", max_value)
        if df is not None:
            df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, f"metrics_{self.num_training_epochs_completed}.csv"), index=False)

    def evaluate(self,
                 data_iterator,
                 dataset_type="train",
                 num_batches_train=0,
                 save_images=True,
                 metrics=[],
                 save_policies=("TEST_TOP_128", "TEST_BOTTOM_128",
                                "TRAIN_TOP_128", "TRAIN_BOTTOM_128",
                                "VAL_TOP_128", "VAL_BOTTOM_128")
                 ):
        if metrics is None or len(metrics) == 0:
            metrics = self.metrics_to_compute
        if num_batches_train == 0:
            num_batches_train = self.exp_config.BATCH_SIZE
        print(
            f"Running evaluation after epoch:{self.num_training_epochs_completed} and step:{self.num_steps_completed} ")
        labels_predicted = None
        z = None
        mu = None
        sigma = None
        labels = None
        batch_no = 1
        data_iterator.reset_counter(dataset_type)
        reconstruction_losses = []
        retention_policies: List[RetentionPolicy] = list()
        if save_images:
            for policy in save_policies:
                if dataset_type.upper() == policy.split("_")[0]:
                    policy_type = policy.split("_")[1]
                    if "reconstruction_loss" in metrics:
                        rp = RetentionPolicy(dataset_type.upper(),
                                             policy_type=policy_type,
                                             N=int(policy.split("_")[2])
                                             )
                        retention_policies.append(rp)
        while data_iterator.has_next(dataset_type):
            batch_images, batch_labels, manual_labels = data_iterator.get_next_batch(dataset_type)
            # skip last batch
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break
            batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

            reconstructed_image, summary, mu_for_batch, sigma_for_batch, z_for_batch, y_pred, nll, nll_batch = self.sess.run(
                [self.out,
                 self.merged_summary_op,
                 self.mu,
                 self.sigma,
                 self.z,
                 self.y_pred,
                 self.neg_loglikelihood,
                 self.marginal_likelihood
                 ],
                feed_dict={
                    self.inputs: batch_images,
                    self.labels: manual_labels[
                                 :,
                                 :10],
                    self.is_manual_annotated: manual_labels[
                                              :,
                                              10],
                    self.standard_normal: batch_z})
            nll_batch = -nll_batch
            if len(nll_batch.shape) == 0:
                data_iterator.reset_counter(dataset_type)
                print(f"Skipping batch {batch_no}. Investigate and fix this issue later")
                print(
                    f"Length of batch_images: {batch_images.shape} Nll_batch shape: {nll_batch.shape} Nll shape: {nll.shape} Nll:{nll} ")
                break
            if len(nll_batch.shape) != 2:
                raise Exception(f"Shape of nll_batch {nll_batch.shape}")

            """
            Update priority queues for keeping top and bottom N samples for all the required metrics present save_policy
            """
            if save_images:
                try:
                    for rp in retention_policies:
                        rp.update_heap(cost=nll_batch,
                                       exp_config=self.exp_config,
                                       data=[reconstructed_image, np.argmax(batch_labels, axis=1), nll_batch])
                except:
                    print(f"Shape of mse is {nll_batch.shape}")
                    traceback.print_exc()

            labels_predicted_for_batch = np.argmax(softmax(y_pred), axis=1)
            labels_for_batch = np.argmax(batch_labels, axis=1)
            reconstruction_losses.append(nll)

            if labels_predicted is None:
                labels_predicted = labels_predicted_for_batch
                labels = labels_for_batch
            else:
                labels_predicted = np.hstack([labels_predicted, labels_predicted_for_batch])
                labels = np.hstack([labels, labels_for_batch])

            if self.exp_config.return_latent_vector:
                if z is None:
                    mu = mu_for_batch
                    sigma = sigma_for_batch
                    z = z_for_batch
                else:
                    mu = np.vstack([mu, mu_for_batch])
                    sigma = np.vstack([sigma, sigma_for_batch])
                    z = np.vstack([z, z_for_batch])
            batch_no += 1
        print(f"Number of evaluation batches completed {batch_no}")
        print(f"epoch:{self.num_training_epochs_completed} step:{self.num_steps_completed}")
        if "reconstruction_loss" in self.metrics_to_compute:
            reconstruction_loss = mean(reconstruction_losses)
            self.metrics[dataset_type]["reconstruction_loss"].append(
                [self.num_training_epochs_completed, reconstruction_loss])
        if "accuracy" in self.metrics_to_compute:
            accuracy = accuracy_score(labels, labels_predicted)
            self.metrics[dataset_type]["accuracy"].append([self.num_training_epochs_completed, accuracy])

        if save_images:
            self.save_sample_reconstructed_images(dataset_type, retention_policies)

        data_iterator.reset_counter(dataset_type)
        encoded_df = pd.DataFrame(np.transpose(np.vstack([labels, labels_predicted])),
                                  columns=["label", "label_predicted"])

        if self.exp_config.return_latent_vector:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(self.exp_config.Z_DIM)
            # encoded_df[mean_col_names] = mu
            for i, mean_col_name in enumerate(mean_col_names):
                encoded_df[mean_col_name] = mu[:, i]

            for i, sigma_col_name in enumerate(sigma_col_names):
                encoded_df[sigma_col_name] = sigma[:, i]

            for i, z_col_name in enumerate(z_col_names):
                encoded_df[z_col_name] = z[:, i]

        if self.exp_config.write_predictions:
            output_csv_file = get_encoded_csv_file(self.exp_config,
                                                   self.num_training_epochs_completed,
                                                   dataset_type
                                                   )
            print("Saving evaluation results to ", self.exp_config.ANALYSIS_PATH)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)

        return encoded_df

    def encode(self, images):
        mu, sigma, z, y_pred = self.sess.run([self.mu, self.sigma, self.z, self.y_pred],
                                             feed_dict={self.inputs: images})
        return mu, sigma, z, y_pred

    def classify(self, images):
        logits = self.sess.run([self.y_pred],
                               feed_dict={self.inputs: images})

        return logits

    def encode_and_get_features(self, images: np.ndarray):
        mu, sigma, z, dense2_en, reshaped, final_conv = self.sess.run([self.mu,
                                                                       self.sigma,
                                                                       self.z,
                                                                       self.dense2_en,
                                                                       self.reshaped_en,
                                                                       self.final_conv
                                                                       ],
                                                                      feed_dict={self.inputs: images})

        return (mu, sigma, z, dense2_en, reshaped, final_conv)
