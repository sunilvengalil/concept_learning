# -*- coding: utf-8 -*-
from __future__ import division

import json
import math
import traceback
from typing import List, Tuple
import os
import numpy as np
import pandas as pd

from clearn.config import ExperimentConfig
from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.models.architectures.custom.tensorflow_graphs import cnn_n_layer, deconv_n_layer, fcnn_n_layer, \
    fully_deconv_n_layer
from clearn.models.generative_model import GenerativeModel
from clearn.utils import prior_factory as prior
from clearn.utils.retention_policy.policy import RetentionPolicy
from clearn.utils.utils import save_image, get_latent_vector_column, get_padding_info
from clearn.utils.dir_utils import get_eval_result_dir, check_and_create_folder

import tensorflow as tf


class VAE(GenerativeModel):
    _model_name_ = "VAE"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 dao: IDao,
                 test_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 ):
        super().__init__(exp_config, sess, epoch, dao=dao, test_data_iterator=test_data_iterator)
        self.padding_added_row, self.padding_added_col, self.image_sizes = get_padding_info(exp_config,
                                                                                            dao.image_shape
                                                                                            )
        self.metrics_to_compute = ["reconstruction_loss"]
        self.metrics = dict()
        self.metrics[VAE.dataset_type_train] = dict()
        self.metrics[VAE.dataset_type_test] = dict()
        self.metrics[VAE.dataset_type_val] = dict()

        for metric in self.metrics_to_compute:
            self.metrics[VAE.dataset_type_train][metric] = []
            self.metrics[VAE.dataset_type_val][metric] = []
            self.metrics[VAE.dataset_type_test][metric] = []

        # test
        self.sample_num = 64  # number of generated images to be saved
        self.num_images_per_row = 4  # should be a factor of sample_num
        self.label_dim = dao.num_classes  # one hot encoding for 10 classes
        self.mu = tf.compat.v1.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM], name='mu')
        self.sigma = tf.compat.v1.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM],
                                              name='sigma')
        self.images = None
        self._build_model()
        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()
        self.sample_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)
        self.counter, self.start_batch_id, self.start_epoch = self._initialize(read_from_existing_checkpoint,
                                                                               check_point_epochs)
        self.num_training_epochs_completed = self.start_epoch
        self.num_steps_completed = self.start_batch_id

    #   Gaussian Encoder
    def _encoder(self, x, reuse=False):
        print("Encoding")
        if self.exp_config.fully_convolutional:
            gaussian_params = fcnn_n_layer(self, x, self.exp_config.num_units, 2, reuse)
        else:
            gaussian_params = cnn_n_layer(self, x, 2 * self.exp_config.Z_DIM, reuse)
        # The mean parameter is unconstrained

        mean = gaussian_params[:, :self.exp_config.Z_DIM]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.exp_config.Z_DIM:])
        return mean, stddev

    # Bernoulli decoder
    def _decoder(self, z, reuse=False):
        print("Decoding")
        if self.exp_config.fully_convolutional:
            out = fully_deconv_n_layer(self,
                                       z,
                                       self.exp_config.num_units,
                                       self.dao.image_shape[2],
                                       1,
                                       reuse)
        else:
            out = deconv_n_layer(self, z, self.dao.image_shape[2], reuse)
        return out

    def inference(self):
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        self.images = self._decoder(z, reuse=True)

    def _build_model(self):
        # some parameters
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + list(image_dims), name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        # self.standard_normal = tf.compat.v1.placeholder(tf.float32, [bs, self.exp_config.Z_DIM], name='z')

        """ Encode the input """
        self.mu, self.sigma = self._encoder(self.inputs, reuse=False)

        # sampling by re-parameterization technique
        self.z = self.mu + self.sigma * tf.random.normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out = self._decoder(self.z, reuse=False)

        # loss
        print("Activation output layer", self.exp_config.activation_output_layer)
        if self.exp_config.activation_output_layer == "SIGMOID":
            self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

            self.marginal_likelihood = tf.reduce_sum(self.inputs * tf.math.log(self.out) +
                                                     self.exp_config.class_weight * (1 - self.inputs) * tf.math.log(
                1 - self.out),
                                                     [1, 2],
                                                     )
        else:
            # Linear activation
            self.out = out
            mll = tf.compat.v1.losses.mean_squared_error(self.inputs,
                                                         self.out,
                                                         reduction=tf.compat.v1.losses.Reduction.NONE
                                                         )
            print("Mll", mll.shape)
            self.marginal_likelihood = -tf.compat.v1.reduce_mean(mll, axis=(1, 2, 3))
            print("after reduction", self.marginal_likelihood.shape)
        self.neg_loglikelihood = -tf.reduce_mean(self.marginal_likelihood)

        kl = 0.5 * tf.reduce_sum(tf.square(self.mu) +
                                 tf.square(self.sigma) -
                                 tf.math.log(1e-8 + tf.square(self.sigma)) - 1, [1])

        self.KL_divergence = tf.reduce_mean(kl)
        self.compute_and_optimize_loss()

    def compute_and_optimize_loss(self):
        # evidence_lower_bound = -self.neg_loglikelihood - self.exp_config.beta * self.KL_divergence
        self.loss = self.neg_loglikelihood + self.exp_config.beta * self.KL_divergence

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.exp_config.learning_rate, beta1=self.exp_config.beta1_adam) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """

        # for test
        # self.fake_images = self._decoder(self.standard_normal, reuse=True)

        """ Summary """
        tf.summary.scalar("Negative Log Likelihood", self.neg_loglikelihood)
        tf.summary.scalar("K L Divergence", self.KL_divergence)
        tf.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def get_trainable_vars(self):
        return tf.trainable_variables()

    def train(self, train_val_data_iterator):
        start_batch_id = self.start_batch_id
        start_epoch = self.start_epoch
        self.num_batches_train = train_val_data_iterator.get_num_samples("train") // self.exp_config.BATCH_SIZE
        evaluation_run_for_last_epoch = False
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            mean_nll = 0
            num_samples_processed = 0
            for batch in range(start_batch_id, self.num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images, _, manual_labels, _ = train_val_data_iterator.get_next_batch("train")
                if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                    break
                batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss, marginal_ll = self.sess.run([self.optim,
                                                                                      self.merged_summary_op,
                                                                                      self.loss,
                                                                                      self.neg_loglikelihood,
                                                                                      self.KL_divergence,
                                                                                      self.marginal_likelihood
                                                                                      ],
                                                                                     feed_dict={
                                                                                         self.inputs: batch_images
                                                                                     })
                marginal_ll = -marginal_ll
                sum_nll_batch = np.mean(marginal_ll) * self.exp_config.BATCH_SIZE
                mean_nll = (mean_nll * num_samples_processed + sum_nll_batch) / (
                        num_samples_processed + self.exp_config.BATCH_SIZE)
                self.counter += 1
                self.num_steps_completed = batch + 1
                num_samples_processed = self.num_steps_completed * self.exp_config.BATCH_SIZE
                # print(f"Epoch:{epoch} Batch:{batch}  loss={loss} nll={nll_loss} kl_loss={kl_loss} batch_mean_nll={np.mean(marginal_ll)}  overall_mean_nll={mean_nll} Number of samples completed ={num_samples_processed}")

                # self.writer.add_summary(summary_str, self.counter - 1)
            self.num_training_epochs_completed = epoch + 1
            print(f"Completed {self.num_training_epochs_completed} epochs")
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
                        print(f"{metric}: train: {self.metrics[VAE.dataset_type_train][metric][-1]}")
                        print(f"{metric}: val: {self.metrics[VAE.dataset_type_val][metric][-1]}")
                        print(f"{metric}: test: {self.metrics[VAE.dataset_type_test][metric][-1]}")
                    self.save_metrics()
                    evaluation_run_for_last_epoch = True

            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")

            if self.num_batches_train > start_batch_id:
                print(f"Epoch:{epoch}   loss={loss} nll={nll_loss} kl_loss={kl_loss}")
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

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

        # save metrics
        self.save_metrics()

    def save_metrics(self):
        df = None
        for metric in self.metrics_to_compute:
            if len(self.metrics["train"][metric]) == 0:
                continue
            df = pd.DataFrame(np.asarray(self.metrics["train"][metric])[:, 0], columns=["epoch"])
            df[f"train_{metric}_mean"] = np.asarray(self.metrics["train"][metric])[:, 1]
            df[f"val_{metric}_mean"] = np.asarray(self.metrics["val"][metric])[:, 1]
            df[f"test_{metric}_mean"] = np.asarray(self.metrics["test"][metric])[:, 1]
            if np.asarray(self.metrics["val"][metric]).shape[1] == 3:
                df[f"train_{metric}_std"] = np.asarray(self.metrics["train"][metric])[:, 2]
                df[f"val_{metric}_std"] = np.asarray(self.metrics["val"][metric])[:, 2]
                df[f"test_{metric}_std"] = np.asarray(self.metrics["test"][metric])[:, 2]

            df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, f"{metric}_{self.start_epoch}.csv"),
                      index=False)
            max_value = df[f"test_{metric}_mean"].max()
            print(f"Max test {metric}", max_value)
            min_value = df[f"test_{metric}_mean"].min()
            print(f"Min test {metric}", min_value)

    def evaluate(self, data_iterator, dataset_type="train", num_batches_train=0, save_images=True,
                 metrics=[],
                 save_policies=("TEST_TOP_128", "TEST_BOTTOM_128",
                                "TRAIN_TOP_128", "TRAIN_BOTTOM_128",
                                "VAL_TOP_128", "VAL_BOTTOM_128")
                 ):
        if metrics is None or len(metrics) == 0:
            metrics = self.metrics_to_compute
        print(
            f"Running evaluation after epoch:{self.num_training_epochs_completed} and step:{self.num_steps_completed} ")
        start_eval_batch = 0
        num_eval_batches = data_iterator.get_num_samples(dataset_type) // self.exp_config.BATCH_SIZE
        mu = None
        sigma = None
        z = None
        data_iterator.reset_counter(dataset_type)
        reconstruction_losses = None
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
        for batch_no in range(start_eval_batch, num_eval_batches):
            batch_images, batch_labels, manual_labels, _ = data_iterator.get_next_batch(dataset_type)
            # skip last batch
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break

            reconstructed_image, summary, mu_for_batch, sigma_for_batch, z_for_batch, nll, nll_batch = self.sess.run(
                [self.out,
                 self.merged_summary_op,
                 self.mu,
                 self.sigma,
                 self.z,
                 self.neg_loglikelihood,
                 self.marginal_likelihood
                 ],
                feed_dict={
                    self.inputs: batch_images
                })
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
            if reconstruction_losses is None:
                reconstruction_losses = nll_batch
            else:
                reconstruction_losses = np.hstack([reconstruction_losses, nll_batch])
            if self.exp_config.return_latent_vector:
                if z is None:
                    mu = mu_for_batch
                    sigma = sigma_for_batch
                    z = z_for_batch
                else:
                    mu = np.vstack([mu, mu_for_batch])
                    sigma = np.hstack([sigma, sigma_for_batch])
                    z = np.vstack([z, z_for_batch])
        encoded_df = None
        if reconstruction_losses is not None and reconstruction_losses.shape[0] > 0:
            if "reconstruction_loss" in self.metrics_to_compute:
                reconstruction_loss = np.mean(reconstruction_losses)
                self.metrics[dataset_type]["reconstruction_loss"].append(
                    [self.num_training_epochs_completed, reconstruction_loss, np.std(reconstruction_losses)])
                encoded_df = pd.DataFrame(reconstruction_losses,
                                          columns=["reconstruction_loss"])

        if save_images:
            self.save_sample_reconstructed_images(dataset_type, retention_policies)

        data_iterator.reset_counter(dataset_type)

        if self.exp_config.return_latent_vector and mu is not None and mu.shape[0] > 0:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(self.exp_config.Z_DIM)
            if encoded_df is None:
                encoded_df = pd.DataFrame(mu, columns=mean_col_names)
            else:
                encoded_df[mean_col_names] = mu
            for i, sigma_col_name in enumerate(sigma_col_names):
                encoded_df[sigma_col_name] = sigma[:, i]

            for i, z_col_name in enumerate(z_col_names):
                encoded_df[z_col_name] = z[:, i]
        if self.exp_config.write_predictions and encoded_df is not None:
            output_csv_file = get_encoded_csv_file(self.exp_config,
                                                   self.num_training_epochs_completed,
                                                   dataset_type
                                                   )
            print("Saving evaluation results to ", self.exp_config.ANALYSIS_PATH)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)

        return encoded_df

    def save_sample_reconstructed_images(self, dataset_type, retention_policies, class_label=None):
        reconstructed_dir = get_eval_result_dir(self.exp_config.PREDICTION_RESULTS_PATH,
                                                self.num_training_epochs_completed,
                                                self.num_steps_completed)
        if class_label is not None:
            reconstructed_dir = check_and_create_folder(reconstructed_dir + f"class_{class_label}/")

        for rp in retention_policies:
            if rp.size() == 0:
                continue
            num_samples_per_image = min(64, rp.size())
            manifold_w = 4
            manifold_h = math.ceil(num_samples_per_image / manifold_w)

            num_images = rp.size() // num_samples_per_image
            if dataset_type.upper() == rp.data_type.upper():
                for image_no in range(num_images):
                    file_image = f"{dataset_type}_{rp.policy_type}_{image_no}.png"
                    original_image_filename = f"orig_{dataset_type}_{rp.policy_type}_{image_no}.png"
                    file_label = f"{dataset_type}_{rp.policy_type}_{image_no}_labels.json"
                    file_loss = f"{dataset_type}_{rp.policy_type}_{image_no}_loss.json"
                    samples_to_save = np.zeros((num_samples_per_image,
                                                self.dao.image_shape[0],
                                                self.dao.image_shape[1],
                                                self.dao.image_shape[2]))
                    original_image = np.zeros((num_samples_per_image,
                                               self.dao.image_shape[0],
                                               self.dao.image_shape[1],
                                               self.dao.image_shape[2]))

                    labels = np.zeros(num_samples_per_image)
                    losses = np.zeros(num_samples_per_image)
                    for sample_num, e in enumerate(
                            rp.data_queue[image_no * num_samples_per_image: (image_no + 1) * num_samples_per_image]):
                        samples_to_save[sample_num, :, :, :] = e[2][0]
                        labels[sample_num] = e[2][1]
                        losses[sample_num] = e[2][2]
                        original_image[sample_num, :, :, :] = e[2][3]
                    save_image(samples_to_save,
                               [manifold_h, manifold_w],
                               reconstructed_dir + file_image,
                               normalize=False)
                    # print(f"Saving original image  to {reconstructed_dir + original_image_filename}")
                    save_image(original_image,
                               [manifold_h, manifold_w],
                               reconstructed_dir + original_image_filename)

                    with open(reconstructed_dir + file_label, "w") as fp:
                        json.dump(labels.tolist(),
                                  fp)

                    with open(reconstructed_dir + file_loss, "w") as fp:
                        json.dump(losses.tolist(),
                                  fp)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self._model_name_, self.exp_config.dataset_name,
            self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

    def encode(self, images):
        mu, sigma, z = self.sess.run([self.mu, self.sigma, self.z],
                                     feed_dict={self.inputs: images})
        return mu, sigma, z

    def get_decoder_weights_bias(self):
        if self.exp_config.fully_convolutional:
            num_deconv_layers = len(self.exp_config.num_units)
        else:
            num_deconv_layers = len(self.exp_config.num_units) - self.exp_config.num_dense_layers
        param_names = []
        for layer_num in range(num_deconv_layers):
            param_names.append(f"decoder/de_conv_{layer_num}/w:0")
            param_names.append(f"decoder/de_conv_{layer_num}/biases:0")
        param_names.append("decoder/de_out/w:0")
        param_names.append("decoder/de_out/biases:0")

        default_graph = tf.get_default_graph()
        params = [default_graph.get_tensor_by_name(tn) for tn in param_names]
        param_values = self.sess.run(params)
        return {tn: tv for tn, tv in zip(param_names, param_values)}

    def get_encoder_weights_bias(self):
        name_w_1 = "encoder/en_conv1/w:0"
        name_w_2 = "encoder/en_conv2/w:0"
        name_w_3 = "encoder/en_fc3/Matrix:0"
        name_w_4 = "encoder/en_fc4/Matrix:0"

        name_b_1 = "encoder/en_conv1/biases:0"
        name_b_2 = "encoder/en_conv2/biases:0"
        name_b_3 = "encoder/en_fc3/bias:0"
        name_b_4 = "encoder/en_fc4/bias:0"

        layer_param_names = [name_w_1,
                             name_b_1,
                             name_w_2,
                             name_b_2,
                             name_w_3,
                             name_b_3,
                             name_w_4,
                             name_b_4
                             ]

        default_graph = tf.get_default_graph()
        params = [default_graph.get_tensor_by_name(tn) for tn in layer_param_names]
        param_values = self.sess.run(params)
        return {tn: tv for tn, tv in zip(layer_param_names, param_values)}

    def get_encoder_features_list(self):
        feature_list = []
        feature_names = []
        for key, value in self.encoder_dict.items():
            feature_names.append(key)
            feature_list.append(value)
        return feature_names, feature_list

    def encode_and_get_features(self, images: np.ndarray):
        features_list = [self.mu, self.sigma, self.z]
        hidden_feature_names, hidden_features = self.get_encoder_features_list()
        features_list.extend(hidden_features)

        encoded_features = self.sess.run(features_list,
                                         feed_dict={self.inputs: images
                                                    })

        return hidden_feature_names, encoded_features[0], encoded_features[1], encoded_features[2], encoded_features[3:]

    def get_decoder_features_list(self):
        feature_list = []
        feature_names = []
        for key, value in self.decoder_dict.items():
            feature_names.append(key)
            feature_list.append(value)
        return feature_names, feature_list

    def decode_and_get_features(self, z: np.ndarray, layer_num=None, feature_num=None):
        features_list = [self.out]
        hidden_feature_names, hidden_features = self.get_decoder_features_list()
        features_list.extend(hidden_features)
        decoded_features = self.sess.run(features_list,
                                         feed_dict={self.z: z
                                                    }
                                         )
        if layer_num is not None:
            for decoded_feature, f in zip(decoded_features[1:], hidden_feature_names):
                if feature_num is not None:
                    if str(layer_num) in f:
                        if isinstance(feature_num, Tuple) or isinstance(feature_num, List):
                            return [f], (decoded_features[0], decoded_feature[:, :, :, feature_num[0]:feature_num[1]])
                        else:
                            return [f], (decoded_features[0], decoded_feature[:, :, :, feature_num])
                else:
                    return [f], (decoded_features[0], decoded_feature)
        else:
            return hidden_feature_names, decoded_features

    def decode(self, z):
        images = self.sess.run(self.out, feed_dict={self.z: z})
        return images

    def load_from_checkpoint(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

        # restore check-point if it exits
        could_load, checkpoint_counter = self._load(self.exp_config.TRAINED_MODELS_PATH)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        return checkpoint_counter
