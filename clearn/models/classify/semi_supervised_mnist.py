# -*- coding: utf-8 -*-
from __future__ import division

import traceback
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from statistics import mean

from clearn.config import ExperimentConfig
from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.models.classify.classifier import ClassifierModel
from clearn.models.vae import VAE
from clearn.utils.retention_policy.policy import RetentionPolicy
from clearn.utils.utils import get_latent_vector_column, get_padding_info, save_images
from scipy.special import softmax
from sklearn.metrics import accuracy_score

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.compat.v1 import placeholder
from clearn.utils.tensorflow_wrappers import linear, conv2d


class SemiSupervisedClassifierMnist(VAE):
    _model_name_ = "SemiSupervisedClassifierMnist"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 dao: IDao,
                 test_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 num_individual_samples_annotated=0,
                 num_samples_wrongly_annotated=0,
                 total_confidence_of_wrong_annotation=0,
                 ):
        # Whether the sample was manually annotated.
        self.is_manual_annotated = placeholder(tf.float32, [exp_config.BATCH_SIZE], name="is_manual_annotated")
        self.labels = placeholder(tf.float32, [exp_config.BATCH_SIZE, dao.num_classes], name='manual_label')
        self.padding_added_row, self.padding_added_col, self.image_sizes = get_padding_info(exp_config,
                                                                                            dao.image_shape)
        self.num_individual_samples_annotated = num_individual_samples_annotated
        self.num_samples_wrongly_annotated = num_samples_wrongly_annotated
        self.total_confidence_of_wrong_annotation = total_confidence_of_wrong_annotation
        concept_dict = exp_config.concept_dict

        if exp_config.fully_convolutional and concept_dict is not None and len(concept_dict) > 0:
            latent_image_dim = self.image_sizes[len(exp_config.num_units)]
            self.concepts_stride = 1

            if latent_image_dim[0] % self.concepts_stride == 0:
                self.num_concpets_per_row = latent_image_dim[0] // self.concepts_stride
            else:
                self.num_concpets_per_row = (latent_image_dim[0] // self.concepts_stride) + 1
            if latent_image_dim[1] % self.concepts_stride == 0:
                self.num_concpets_per_col = latent_image_dim[1] // self.concepts_stride
            else:
                self.num_concpets_per_col = (latent_image_dim[1] // self.concepts_stride) + 1

            # self.is_concepts_annotated = placeholder(tf.float32,
            #                                          [exp_config.BATCH_SIZE,
            #                                           self.num_concpets_per_row,
            #                                           self.num_concpets_per_col],
            #                                          name="is_concepts_annotated")
            # self.concepts_labels = placeholder(tf.float32,
            #                                    [exp_config.BATCH_SIZE,
            #                                     self.num_concpets_per_row,
            #                                     self.num_concpets_per_col,
            #                                     exp_config.dao.num_classes],
            #                                    name='manual_label_concepts')
        if exp_config.concept_dict is not None and len(exp_config.concept_dict) > 0:
            self.layers_to_apply_concept_loss = []
            self.unique_concepts: Dict[int, List] = dict()
            self.mask_for_concept_no = dict()
            for layer_num in exp_config.concept_dict.keys():
                self.unique_concepts[layer_num] = concept_dict[layer_num]["unique_concepts"]
                self.mask_for_concept_no[layer_num] = dict()
                for concept_no in self.unique_concepts[layer_num]:
                    self.mask_for_concept_no[layer_num][concept_no] = placeholder(tf.float32, exp_config.BATCH_SIZE)

        super().__init__(exp_config=exp_config,
                         sess=sess,
                         epoch=epoch,
                         dao=dao,
                         test_data_iterator=test_data_iterator,
                         read_from_existing_checkpoint=read_from_existing_checkpoint,
                         check_point_epochs=check_point_epochs)

        self.metrics_to_compute = ["accuracy", "reconstruction_loss"]
        self.metrics = dict()
        self.metrics[SemiSupervisedClassifierMnist.dataset_type_train] = dict()
        self.metrics[SemiSupervisedClassifierMnist.dataset_type_test] = dict()
        self.metrics[SemiSupervisedClassifierMnist.dataset_type_val] = dict()
        for metric in self.metrics_to_compute:
            self.metrics[SemiSupervisedClassifierMnist.dataset_type_train][metric] = []
            self.metrics[SemiSupervisedClassifierMnist.dataset_type_val][metric] = []
            self.metrics[SemiSupervisedClassifierMnist.dataset_type_test][metric] = []

    def compute_and_optimize_loss(self):
        if self.exp_config.fully_convolutional:
            concepts_stride = 1
            z_reshaped = tf.reshape(self.z, [self.exp_config.BATCH_SIZE,
                                             self.image_sizes[len(self.exp_config.num_units)][0],
                                             self.image_sizes[len(self.exp_config.num_units)][0],
                                             1
                                             ]
                                    )
            self.concepts_pred = conv2d(z_reshaped,
                                        self.exp_config.dao.num_classes,
                                        k_h=2,
                                        k_w=2,
                                        d_h=concepts_stride,
                                        d_w=concepts_stride,
                                        stddev=0.02,
                                        name="predict_concepts")

        if self.exp_config.fully_convolutional:
            self.supervised_loss_concepts = 0
            self.supervised_loss_concepts_per_layer = dict()
            if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                for layer_num in list(self.exp_config.concept_dict.keys()):
                    if layer_num >= len(self.exp_config.num_units) + 1:
                        continue
                    decoder_feature = f"de_conv_{layer_num}"
                    print("layer_num", layer_num, decoder_feature)
                    f = self.decoder_dict[decoder_feature]
                    print(f.shape)
                    num_concepts = len(self.exp_config.concept_dict[layer_num]["unique_concepts"])
                    self.supervised_loss_concepts_per_layer[layer_num] = dict()
                    self.mse_for_all_images = dict()
                    self.mse_for_all_images_masked = dict()
                    for concept_no in self.unique_concepts[layer_num]:
                        # print("feature shape", f.shape)
                        # print(f"Computing loss for {layer_num} concept {concept_no}")
                        input_resized = tf.image.resize(self.inputs, [f.shape[1], f.shape[2]],
                                                        preserve_aspect_ratio=True)
                        #print("input shape ", input_resized.shape)
                        mse = tf.compat.v1.losses.mean_squared_error(f[:, :, :, concept_no:concept_no + 1],
                                                                     input_resized,
                                                                     reduction=tf.compat.v1.losses.Reduction.NONE
                                                                     )
                        self.mse_for_all_images[concept_no] = tf.compat.v1.reduce_mean(mse, axis=(1, 2, 3))
                        self.mse_for_all_images_masked[concept_no] = tf.math.multiply(
                            self.mse_for_all_images[concept_no], self.mask_for_concept_no[layer_num][concept_no])

                        self.supervised_loss_concepts_per_layer[layer_num][concept_no] =  tf.math.divide_no_nan(tf.compat.v1.reduce_sum(
                            self.mse_for_all_images_masked[concept_no]), tf.compat.v1.reduce_sum(self.mask_for_concept_no[layer_num][concept_no]))

                        self.supervised_loss_concepts += self.supervised_loss_concepts_per_layer[layer_num][concept_no]

                        # Make response for other images zero
                        # mse_other_images = tf.compat.v1.losses.mean_squared_error(f[:, :, :, concept_no:concept_no + 1],
                        #                                                           tf.zeros_like(f[:, :, :, concept_no:concept_no + 1]),
                        #                                                           reduction=tf.compat.v1.losses.Reduction.NONE)
                        # print("Shape mse_other_images", mse_other_images.shape)
                        # inverted_mask = tf.math.subtract(tf.ones_like(self.mask_for_concept_no[layer_num][concept_no]),
                        #                             self.mask_for_concept_no[layer_num][concept_no])
                        # mse_for_other_images_masked= tf.math.multiply(tf.compat.v1.reduce_mean(mse_other_images,axis=(1, 2, 3)),
                        #                                               inverted_mask)
                        # supervised_loss_concepts_per_layer_other = tf.math.divide_no_nan(tf.compat.v1.reduce_sum(mse_for_other_images_masked),
                        #                                                                                         tf.compat.v1.reduce_sum(inverted_mask))
                        # self.supervised_loss_concepts += supervised_loss_concepts_per_layer_other
                        # Make sure all other feature maps are zero for this activation
                        # mse_other_layers = tf.compat.v1.losses.mean_squared_error(f[:, :, :, 0:concept_no],
                        #                                                           tf.zeros_like(f[:, :, :, 0:concept_no]),
                        #                                                           reduction=tf.compat.v1.losses.Reduction.NONE
                        #                                                           )
                        #
                        # self.mse_for_all_images[concept_no] = tf.compat.v1.reduce_mean(mse_other_layers, axis=(1, 2, 3))

        self.y_pred = linear(self.z, self.dao.num_classes)
        self.supervised_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                                         logits=self.y_pred,
                                                                         weights=self.is_manual_annotated
                                                                         )

        if self.exp_config.fully_convolutional:
            self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood + \
                        self.exp_config.beta * self.KL_divergence + \
                        self.exp_config.supervise_weight * self.supervised_loss + \
                        self.exp_config.supervise_weight_concepts * self.supervised_loss_concepts
        else:
            self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood + \
                        self.exp_config.beta * self.KL_divergence + \
                        self.exp_config.supervise_weight * self.supervised_loss

        # if self.exp_config.uncorrelated_features:
        # last_feature = list(self.decoder_dict.keys())[-1]
        #     f = self.decoder_dict[last_feature]
        #     identity = tf.eye(num_rows=int(f.shape[3]),num_columns=int(f.shape[3]), batch_shape=[int(f.shape[0])], dtype=tf.float32)
        #     f = tf.reshape(f, [-1, int(f.shape[1]) * int(f.shape[2]), int(f.shape[3]) ])
        #     corr = tfp.stats.correlation(f, sample_axis=1)
        #
        #     self.corr_loss = tf.norm(corr - identity)
        #     self.loss = self.loss + self.corr_loss

        """ Training """
        # optimizers
        t_vars = tf.compat.v1.trainable_variables()
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optim = tf.compat.v1.train.AdamOptimizer(self.exp_config.learning_rate,
                                                          beta1=self.exp_config.beta1_adam) \
                .minimize(self.loss, var_list=t_vars)

        """ Summary """
        tf.compat.v1.summary.scalar("Negative Log Likelihood", self.neg_loglikelihood)
        tf.compat.v1.summary.scalar("K L Divergence", self.KL_divergence)
        tf.compat.v1.summary.scalar("Supervised Loss", self.supervised_loss)

        tf.compat.v1.summary.scalar("Total Loss", self.loss)
        # final summary operations
        self.merged_summary_op = tf.compat.v1.summary.merge_all()

    def train(self, train_val_data_iterator):
        start_batch_id = self.start_batch_id
        start_epoch = self.start_epoch
        self.num_batches_train = train_val_data_iterator.get_num_samples("train") // self.exp_config.BATCH_SIZE
        evaluation_run_for_last_epoch = False

        images_saved = 0
        num_images_to_save = 256
        num_samples_per_image = 64
        manifold_w = 4
        manifold_h = num_samples_per_image // manifold_w

        for epoch in range(start_epoch, self.epoch):
            evaluation_run_for_last_epoch = False
            supervised_loss_concepts_epoch = dict()
            if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                for layer_num in self.exp_config.concept_dict.keys():
                    if layer_num == len(self.exp_config.num_units) + 1:
                        continue
                    supervised_loss_concepts_epoch[layer_num] = []
            for batch in range(start_batch_id, self.num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images, batch_labels, manual_labels, manual_labels_concepts = train_val_data_iterator.get_next_batch(
                    "train")
                labels_categorical = np.argmax(batch_labels, axis=1)
                if num_images_to_save > images_saved:
                    save_images(batch_images[0:64],
                                [manifold_h, manifold_w],
                                self.exp_config.PREDICTION_RESULTS_PATH + "/" + f"train_{batch}.png")
                    images_saved += 64

                if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                    break

                if self.exp_config.fully_convolutional:
                    tensor_list = [self.optim,
                                   self.merged_summary_op,
                                   self.loss,
                                   self.neg_loglikelihood,
                                   self.marginal_likelihood,
                                   self.KL_divergence,
                                   self.supervised_loss]
                    feed_dict = {
                        self.inputs: batch_images,
                        self.labels: manual_labels[
                                     :,
                                     :self.dao.num_classes],
                        self.is_manual_annotated: manual_labels[
                                                  :,
                                                  self.dao.num_classes],
                    }

                    if self.exp_config.uncorrelated_features:
                        tensor_list.append(self.supervised_loss_concepts)
                        tensor_list.append(self.corr_loss)
                        return_list = self.sess.run(tensor_list,
                                                    feed_dict=feed_dict)
                        loss = return_list[2]
                        nll_loss = return_list[3]
                        kl_loss = return_list[4]
                        supervised_loss = return_list[5]
                        supervised_loss_concepts = return_list[6]
                        correlation_loss = return_list[7]
                    else:
                        if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                            for layer_num in self.exp_config.concept_dict.keys():
                                if layer_num < len(self.exp_config.num_units) + 1:
                                    tensor_list.append(self.supervised_loss_concepts_per_layer[layer_num])
                                for concept_no in self.unique_concepts[layer_num]:
                                    masks = np.zeros(self.exp_config.BATCH_SIZE)
                                    if concept_no != -1:
                                        masks[(manual_labels[:, self.dao.num_classes + 1] == layer_num) * (
                                                    labels_categorical == concept_no)] = 1
                                    feed_dict[self.mask_for_concept_no[layer_num][concept_no]] = masks
                        return_list = self.sess.run(tensor_list,
                                                    feed_dict=feed_dict)
                        loss = return_list[2]
                        nll_loss = return_list[3]
                        kl_loss = return_list[4]
                        supervised_loss = return_list[5]

                        supervised_loss_concepts = dict()
                        supervised_loss_concepts_total = dict()
                        if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                            for i, layer_num in enumerate(self.exp_config.concept_dict.keys()):
                                if layer_num >= len(self.exp_config.num_units) + 1:
                                    continue
                                supervised_loss_concepts[layer_num] = return_list[6 + i]
                                supervised_loss_concepts_total[layer_num] = 0
                                for k, v in supervised_loss_concepts[layer_num].items():
                                    supervised_loss_concepts_total[layer_num] += v
                else:
                    feed_dict = {
                        self.inputs: batch_images,
                        self.labels: manual_labels[:, :self.dao.num_classes],
                        self.is_manual_annotated: manual_labels[:, self.dao.num_classes],
                    }
                    if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                        for layer_num in self.exp_config.concept_dict.keys():
                            print(f"Generating mask for layer {layer_num} features {self.unique_concepts[layer_num]} ")
                            for concept_no in self.unique_concepts[layer_num]:
                                masks = np.zeros(self.exp_config.BATCH_SIZE)
                                if concept_no == -1:
                                    # special case for handling samples from the original classes
                                    masks[manual_labels[:, self.dao.num_classes + 1] <= 9] = 1
                                else:
                                    masks[manual_labels[:, self.dao.num_classes + 1] == layer_num] = 1
                                print(
                                    f"Number of samples with gt for layer {layer_num} concept {concept_no} {np.sum(masks)}")
                                feed_dict[self.mask_for_concept_no[layer_num][concept_no]] = masks
                    _, summary_str, loss, nll_loss, nll_batch, kl_loss, supervised_loss = self.sess.run(tensor_list,
                                                                                                        feed_dict
                                                                                                        )
                if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                    for i, layer_num in enumerate(self.exp_config.concept_dict.keys()):
                        if layer_num == len(self.exp_config.num_units ) + 1:
                            continue
                        supervised_loss_concepts_epoch[layer_num].append(supervised_loss_concepts_total[layer_num])
                # if self.exp_config.fully_convolutional:
                #     if self.exp_config.uncorrelated_features:
                #         print(
                #             f"Epoch: {epoch}/{batch}, Nll_loss : {nll_loss} KLD:{kl_loss}  Supervised loss:{supervised_loss} Supervised loss concepts:{supervised_loss_concepts}  ccrrelation loss:{correlation_loss}")
                #     else:
                #         print(f"Epoch: {epoch}/{batch}, Loss:{loss} Nll_loss : {nll_loss} KLD:{kl_loss}  Supervised loss:{supervised_loss} Supervised loss concepts:{supervised_loss_concepts}")
                #
                # else:
                #     print(f"Epoch: {epoch}/{batch}, Nll_loss : {nll_loss} KLD:{kl_loss}  Supervised loss:{supervised_loss} ")
                self.counter += 1
                self.num_steps_completed = batch + 1
                # self.writer.add_summary(summary_str, self.counter - 1)

            print(f"Epoch: {epoch}/{batch}, Nll_loss : {nll_loss}")
            if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                for layer_num in self.exp_config.concept_dict.keys():
                    if layer_num == len(self.exp_config.num_units) + 1:
                        continue
                    print(f"Supervised loss concept Layer {layer_num} {sum(supervised_loss_concepts_epoch[layer_num])}")

            self.num_training_epochs_completed = epoch + 1
            print(f"Completed {epoch} epochs")
            if self.exp_config.run_evaluation_during_training:
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
                        print(f"{metric}: train: {self.metrics[ClassifierModel.dataset_type_train][metric][-1]}")
                        print(f"{metric}: val: {self.metrics[ClassifierModel.dataset_type_val][metric][-1]}")
                        print(f"{metric}: test: {self.metrics[ClassifierModel.dataset_type_test][metric][-1]}")
                    self.save_metrics()
                    evaluation_run_for_last_epoch = True
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
        if not evaluation_run_for_last_epoch:
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

        self.save_metrics()

    def save_metrics(self):
        # save metrics
        df = None
        for i, metric in enumerate(self.metrics_to_compute):
            if len(self.metrics["train"][metric]) == 0:
                continue
            column_name = f"train_{metric}_mean"
            if i == 0:
                train_metric = np.asarray(self.metrics["train"][metric])[:, 0:2]
                df = pd.DataFrame(train_metric, columns=["epoch", column_name])
            else:
                df[column_name] = np.asarray(self.metrics["train"][metric])[:, 1]

            df[f"val_{metric}_mean"] = np.asarray(self.metrics["val"][metric])[:, 1]
            df[f"test_{metric}_mean"] = np.asarray(self.metrics["test"][metric])[:, 1]

            if np.asarray(self.metrics["val"][metric]).shape[1] == 3:
                df[f"train_{metric}_std"] = np.asarray(self.metrics["train"][metric])[:, 2]
                df[f"val_{metric}_std"] = np.asarray(self.metrics["val"][metric])[:, 2]
                df[f"test_{metric}_std"] = np.asarray(self.metrics["test"][metric])[:, 2]

            max_value = df[f"test_{metric}_mean"].max()
            print(f"Max test {metric}_mean", max_value)
            min_value = df[f"test_{metric}_mean"].min()
            print(f"Minimum test {metric}_mean", min_value)

        df["num_individual_samples_annotated"] = self.num_individual_samples_annotated
        df["num_samples_wrongly_annotated"] = self.num_samples_wrongly_annotated
        df["total_confidence_of_wrong_annotation"] = self.total_confidence_of_wrong_annotation

        if df is not None:
            df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, f"metrics_{self.start_epoch}.csv"),
                      index=False)

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
        labels_predicted_proba = None
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
            batch_images, batch_labels, manual_labels, manual_labels_concepts = data_iterator.get_next_batch(
                dataset_type)

            # skip last batch
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break

            if self.exp_config.fully_convolutional:
                feed_dict = {
                    self.inputs: batch_images,
                    self.labels: manual_labels[:, :self.dao.num_classes],
                    self.is_manual_annotated: manual_labels[:, self.dao.num_classes]
                }
                if self.exp_config.concept_dict is not None and len(self.exp_config.concept_dict) > 0:
                    for layer_num in self.exp_config.concept_dict.keys():
                        # print(self.mask_for_concept_no[layer_num])
                        for concept_no in self.unique_concepts[layer_num]:
                            # print("concept number", concept_no)
                            # print(self.mask_for_concept_no[layer_num][concept_no])
                            masks = np.zeros(self.exp_config.BATCH_SIZE)
                            if concept_no == -1:
                                masks[manual_labels[:, self.dao.num_classes + 1] <= 9] = 1
                            else:
                                masks[manual_labels[:, self.dao.num_classes + 1] == layer_num] = 1
                            feed_dict[self.mask_for_concept_no[layer_num][concept_no]] = masks

                return_list = self.sess.run([self.out,
                                             self.merged_summary_op,
                                             self.mu,
                                             self.sigma,
                                             self.z,
                                             self.y_pred,
                                             self.neg_loglikelihood,
                                             self.marginal_likelihood],
                                            feed_dict=feed_dict)
                reconstructed_image = return_list[0]
                mu_for_batch = return_list[2]
                sigma_for_batch = return_list[3]
                z_for_batch = return_list[4]
                y_pred = return_list[5]
                nll = return_list[6]
                nll_batch = return_list[7]
            else:
                return_list = self.sess.run(
                    [self.out,
                     self.merged_summary_op,
                     self.mu,
                     self.sigma,
                     self.z,
                     self.y_pred,
                     self.neg_loglikelihood,
                     self.marginal_likelihood],
                    feed_dict={
                        self.inputs: batch_images,
                        self.labels: manual_labels[:, :10],
                        self.is_manual_annotated: manual_labels[:, 10]
                    })
                reconstructed_image = return_list[0]
                mu_for_batch = return_list[2]
                sigma_for_batch = return_list[3]
                z_for_batch = return_list[4]
                y_pred = return_list[5]
                nll = return_list[6]
                nll_batch = return_list[7]

            nll_batch = -nll_batch
            if len(nll_batch.shape) == 0:
                data_iterator.reset_counter(dataset_type)
                print(f"Skipping batch {batch_no}. Investigate and fix this issue later")
                print(
                    f"Length of batch_images: {batch_images.shape} Nll_batch shape: {nll_batch.shape} Nll shape: {nll.shape} Nll:{nll} ")
                break
            # if len(nll_batch.shape) != 2:
            #     raise Exception(f"Shape of nll_batch {nll_batch.shape}")
            """
            Update priority queues for keeping top and bottom N samples for all the required metrics present save_policy
            """
            if save_images:
                try:
                    for rp in retention_policies:
                        rp.update_heap(cost=nll_batch,
                                       exp_config=self.exp_config,
                                       data=[reconstructed_image, np.argmax(batch_labels, axis=1), nll_batch,
                                             batch_images])
                except:
                    print(f"Shape of mse is {nll_batch.shape}")
                    traceback.print_exc()

            predicted_proba_batch = softmax(y_pred)
            labels_predicted_for_batch = np.argmax(predicted_proba_batch, axis=1)
            labels_for_batch = np.argmax(batch_labels, axis=1)
            reconstruction_losses.append(nll)

            if labels_predicted is None:
                labels_predicted = labels_predicted_for_batch
                labels_predicted_proba = predicted_proba_batch
                labels = labels_for_batch
            else:
                labels_predicted = np.hstack([labels_predicted, labels_predicted_for_batch])
                labels_predicted_proba = np.vstack([labels_predicted_proba, predicted_proba_batch])
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
                [self.num_training_epochs_completed, reconstruction_loss, np.std(reconstruction_losses)])
        if "accuracy" in self.metrics_to_compute:
            accuracy = accuracy_score(labels, labels_predicted)
            self.metrics[dataset_type]["accuracy"].append([self.num_training_epochs_completed, accuracy])

        if save_images:
            self.save_sample_reconstructed_images(dataset_type, retention_policies)

        data_iterator.reset_counter(dataset_type)
        encoded_df = pd.DataFrame(np.transpose(np.vstack([labels, labels_predicted])),
                                  columns=["label", "label_predicted"])

        if self.exp_config.return_latent_vector:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names, predicted_proba_col_names = get_latent_vector_column(
                self.exp_config.Z_DIM, self.dao.num_classes, True)
            # encoded_df[mean_col_names] = mu
            for i, mean_col_name in enumerate(mean_col_names):
                encoded_df[mean_col_name] = mu[:, i]

            for i, sigma_col_name in enumerate(sigma_col_names):
                encoded_df[sigma_col_name] = sigma[:, i]

            for i, z_col_name in enumerate(z_col_names):
                encoded_df[z_col_name] = z[:, i]

            for i, predicted_proba_col_name in enumerate(predicted_proba_col_names):
                encoded_df[predicted_proba_col_name] = labels_predicted_proba[:, i]

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
