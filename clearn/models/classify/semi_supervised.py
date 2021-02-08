# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd

from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.models.architectures.custom.tensorflow_graphs import cnn_4_layer, deconv_4_layer
from clearn.models.vae import VAE
from clearn.utils import prior_factory as prior
from clearn.utils.utils import save_image, save_single_image, get_latent_vector_column
from clearn.utils.dir_utils import get_eval_result_dir
from scipy.special import softmax

import tensorflow as tf
from clearn.utils.tensorflow_wrappers import linear


class SemiSupervisedClassifier(VAE):
    _model_name_ = "SemiSupervisedClassifier"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 dao: IDao,
                 train_val_data_iterator=None,
                 test_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None
                 ):
        # Whether the sample was manually annotated.
        self.is_manual_annotated = tf.placeholder(tf.float32, [exp_config.BATCH_SIZE], name="is_manual_annotated")
        self.labels = tf.placeholder(tf.float32, [exp_config.BATCH_SIZE, dao.num_classes], name='manual_label')
        self.test_data_iterator = test_data_iterator
        super().__init__(exp_config=exp_config,
                         sess=sess,
                         epoch=epoch,
                         dao=dao,
                         train_val_data_iterator=train_val_data_iterator,
                         read_from_existing_checkpoint=read_from_existing_checkpoint,
                         check_point_epochs=check_point_epochs)

    def _encoder(self, x, reuse=False):
        gaussian_params = cnn_4_layer(self, x, 2 * self.exp_config.Z_DIM, reuse)
        # The mean parameter is unconstrained
        mean = gaussian_params[:, :self.exp_config.Z_DIM]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.exp_config.Z_DIM:])

        return mean, stddev

    def decoder(self, z, reuse=False):
        # Models the probability P(X/z)
        return deconv_4_layer(self, z, reuse)

    def get_decoder_weights_bias(self):
        # name_w_1 = "decoder/de_fc1/Matrix:0"
        # name_w_2 = "decoder/de_dc3/w:0"
        # name_w_3 = "decoder/de_dc4/w:0"
        #
        # name_b_1 = "decoder/de_fc1/bias:0"
        # name_b_2 = "decoder/de_dc3/biases:0"
        # name_b_3 = "decoder/de_dc4/biases:0"
        #
        # layer_param_names = [name_w_1,
        #                      name_b_1,
        #                      name_w_2,
        #                      name_b_2,
        #                      name_w_3,
        #                      name_b_3,
        #                      ]
        #
        # default_graph = tf.get_default_graph()
        # params = [default_graph.get_tensor_by_name(tn) for tn in layer_param_names]
        # param_values = self.sess.run(params)
        # return {tn: tv for tn, tv in zip(layer_param_names, param_values)}
        return None

    def get_encoder_weights_bias(self):
        # name_w_1 = "encoder/en_conv1/w:0"
        # name_w_2 = "encoder/en_conv2/w:0"
        # name_w_3 = "encoder/en_fc3/Matrix:0"
        # name_w_4 = "encoder/en_fc4/Matrix:0"
        #
        # name_b_1 = "encoder/en_conv1/biases:0"
        # name_b_2 = "encoder/en_conv2/biases:0"
        # name_b_3 = "encoder/en_fc3/bias:0"
        # name_b_4 = "encoder/en_fc4/bias:0"
        #
        # layer_param_names = [name_w_1,
        #                      name_b_1,
        #                      name_w_2,
        #                      name_b_2,
        #                      name_w_3,
        #                      name_b_3,
        #                      name_w_4,
        #                      name_b_4
        #                      ]
        #
        # default_graph = tf.get_default_graph()
        # params = [default_graph.get_tensor_by_name(tn) for tn in layer_param_names]
        # param_values = self.sess.run(params)
        # return {tn: tv for tn, tv in zip(layer_param_names, param_values)}
        return None

    def compute_and_optimize_loss(self):
        self.y_pred = linear(self.z, 10)
        self.supervised_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                               logits=self.y_pred,
                                                               weights=self.is_manual_annotated
                                                               )

        self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood + \
                    self.exp_config.beta * self.KL_divergence + \
                    self.exp_config.supervise_weight * self.supervised_loss
        # self.loss = -evidence_lower_bound + self.exp_config.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.exp_config.learning_rate, beta1=self.exp_config.beta1_adam) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        self.fake_images = self.decoder(self.standard_normal, reuse=True)

        """ Summary """
        tf.summary.scalar("Negative Log Likelihood", self.neg_loglikelihood)
        tf.summary.scalar("K L Divergence", self.KL_divergence)
        tf.summary.scalar("Supervised Loss", self.supervised_loss)

        tf.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

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
                batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss, supervised_loss = self.sess.run([self.optim,
                                                                                          self.merged_summary_op,
                                                                                          self.loss,
                                                                                          self.neg_loglikelihood,
                                                                                          self.KL_divergence,
                                                                                          self.supervised_loss],
                                                                                         feed_dict={self.inputs: batch_images,
                                                                                                    self.labels: manual_labels[:, :10],
                                                                                                    self.is_manual_annotated: manual_labels[:, 10],
                                                                                                    self.standard_normal: batch_z})
                print("Epoch: [%2d] [%4d/%4d] , loss: %.8f, nll: %.8f, kl: %.8f, supervised_loss: %.4f"
                      % (epoch, batch, self.num_batches_train, loss, nll_loss, kl_loss,
                         supervised_loss))
                self.counter += 1
                self.num_training_epochs_completed = epoch + 1
                self.num_steps_completed = batch + 1
                if self.exp_config.run_evaluation_during_training:
                    if np.mod(batch, self.exp_config.eval_interval) == self.exp_config.eval_interval - 1:
                        train_val_data_iterator.reset_counter("val")
                        self.evaluate(data_iterator=train_val_data_iterator,
                                      dataset_type="val")

                        if self.test_data_iterator is not None:
                            self.evaluate(self.test_data_iterator,
                                          dataset_type="test",
                                          )
                            self.test_data_iterator.reset_counter("test")

                self.writer.add_summary(summary_str, self.counter - 1)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            train_val_data_iterator.reset_counter("train")
            if np.mod(epoch, self.exp_config.model_save_interval) == 0:
                print("Saving check point", self.exp_config.TRAINED_MODELS_PATH)
                self.save(self.exp_config.TRAINED_MODELS_PATH, self.counter)

    def evaluate(self, data_iterator, dataset_type, num_batches_train=0, save_images=True ):
        if num_batches_train == 0:
            num_batches_train = self.num_batches_train
        print(
            f"Running evaluation after epoch:{self.num_training_epochs_completed} and step:{self.num_steps_completed} ")
        start_eval_batch = 0
        reconstructed_images = []
        num_eval_batches = data_iterator.get_num_samples(dataset_type) // self.exp_config.BATCH_SIZE
        manifold_w = 4
        tot_num_samples = min(self.sample_num, self.exp_config.BATCH_SIZE)
        manifold_h = tot_num_samples // manifold_w
        mu = None
        sigma = None
        z = None
        labels = None
        labels_predicted = None
        for batch_no in range(start_eval_batch, num_eval_batches):
            batch_eval_images, batch_labels, manual_labels = data_iterator.get_next_batch(dataset_type)
            if batch_eval_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break
            batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

            reconstructed_image, summary, mu_for_batch, sigma_for_batch, z_for_batch, y_pred = self.sess.run([self.out,
                                                                                                              self.merged_summary_op,
                                                                                                              self.mu,
                                                                                                              self.sigma,
                                                                                                              self.z,
                                                                                                              self.y_pred
                                                                                                              ],
                                                                                                             feed_dict={
                                                                                                                 self.inputs: batch_eval_images,
                                                                                                                 self.labels: manual_labels[
                                                                                                                              :,
                                                                                                                              :10],
                                                                                                                 self.is_manual_annotated: manual_labels[
                                                                                                                                           :,
                                                                                                                                           10],
                                                                                                                 self.standard_normal: batch_z})
            labels_predicted_for_batch = np.argmax(softmax(y_pred), axis=1)
            labels_for_batch = np.argmax(batch_labels, axis=1)

            if labels is None:
                labels_predicted = labels_predicted_for_batch
                labels = labels_for_batch
            else:
                labels_predicted = np.hstack([labels_predicted, labels_predicted_for_batch])
                labels = np.hstack([labels, labels_for_batch])

            if self.exp_config.return_latent_vector:
                if mu is None:
                    mu = mu_for_batch
                    sigma = sigma_for_batch
                    z = z_for_batch
                else:
                    mu = np.vstack([mu, mu_for_batch])
                    sigma = np.vstack([sigma, sigma_for_batch])
                    z = np.vstack([z, z_for_batch])

            training_batch = self.num_training_epochs_completed * num_batches_train + self.num_steps_completed
            if save_images:
                save_single_image(reconstructed_image,
                                  self.exp_config.reconstructed_images_path,
                                  self.num_training_epochs_completed,
                                  self.num_steps_completed,
                                  training_batch,
                                  batch_no,
                                  self.exp_config.BATCH_SIZE)

            # self.writer_v.add_summary(summary, counter)
            reconstructed_images.append(reconstructed_image[:manifold_h * manifold_w, :, :, :])
        print(f"epoch:{self.num_training_epochs_completed} step:{self.num_steps_completed}")
        if save_images:
            reconstructed_dir = get_eval_result_dir(self.exp_config.PREDICTION_RESULTS_PATH,
                                                    self.num_training_epochs_completed,
                                                    self.num_steps_completed)
            for batch_no in range(start_eval_batch, num_eval_batches):
                file = "im_" + str(batch_no) + ".png"
                save_image(reconstructed_images[batch_no], [manifold_h, manifold_w], reconstructed_dir + file)

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
                                                   dataset_type)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)
        print("Evaluation completed")

    def encode(self, images):
        mu, sigma, z, y_pred = self.sess.run([self.mu, self.sigma, self.z],
                                             feed_dict={self.inputs: images})
        return mu, sigma, z, y_pred

    def classify(self, images):
        logits = self.sess.run([self.y_pred],
                               feed_dict={self.inputs: images})

        return logits
