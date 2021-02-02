# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd

from clearn.config import ExperimentConfig
from clearn.config.common_path import get_encoded_csv_file
from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao
from clearn.models.architectures import cnn_3_layer, deconv_3_layer
from clearn.models.generative_model import GenerativeModel
from clearn.utils import prior_factory as prior
from clearn.utils.utils import save_image, save_single_image, get_latent_vector_column
from clearn.utils.dir_utils import get_eval_result_dir

import tensorflow as tf


class VAE(GenerativeModel):
    _model_name_ = "VAE"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 dao: IDao = MnistDao(),
                 ):
        super().__init__(exp_config, sess, epoch)
        self.dao = dao
        # test
        self.sample_num = 64  # number of generated images to be saved
        self.num_images_per_row = 4  # should be a factor of sample_num
        self.label_dim = dao.num_classes  # one hot encoding for 10 classes
        self.mu = tf.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM], name='mu')
        self.sigma = tf.placeholder(tf.float32, [self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM], name='sigma')
        self.images = None
        self._build_model()
        # initialize all variables
        tf.global_variables_initializer().run()
        self.sample_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)
        self.counter, self.start_batch_id, self.start_epoch = self._initialize(train_val_data_iterator,
                                                                               read_from_existing_checkpoint,
                                                                               check_point_epochs)
        self.num_training_epochs_completed = self.start_epoch
        self.num_steps_completed = self.start_batch_id

    #   Gaussian Encoder
    def _encoder(self, x, reuse=False):
        gaussian_params = cnn_3_layer(self, x, 2 * self.exp_config.Z_DIM, reuse)
        # The mean parameter is unconstrained
        mean = gaussian_params[:, :self.exp_config.Z_DIM]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.exp_config.Z_DIM:])
        return mean, stddev

    # Bernoulli decoder
    def decoder(self, z, reuse=False):
        out = deconv_3_layer(self, z, reuse)
        return out

    def inference(self):
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        self.images = self.decoder(z, reuse=True)

    def _build_model(self):
        # some parameters
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
        self.strides = [2, 2, 2, 2, 2]

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        self.standard_normal = tf.placeholder(tf.float32, [bs, self.exp_config.Z_DIM], name='z')


        """ Encode the input """
        # encoding
        self.mu, self.sigma = self._encoder(self.inputs, reuse=False)

        # sampling by re-parameterization technique
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out = self.decoder(self.z, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        if self.exp_config.activation_output_layer == "SIGMOID":
            marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) +
                                                (1 - self.inputs) * tf.log(1 - self.out),
                                                [1, 2])
            self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)

        else:
            #Linear activation
            marginal_likelihood = tf.compat.v1.losses.mean_squared_error(self.inputs, self.out)
            self.neg_loglikelihood = tf.reduce_mean(marginal_likelihood)

        kl = 0.5 * tf.reduce_sum(tf.square(self.mu) +
                                 tf.square(self.sigma) -
                                 tf.log(1e-8 + tf.square(self.sigma)) - 1, [1])

        self.KL_divergence = tf.reduce_mean(kl)

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
        self.fake_images = self.decoder(self.standard_normal, reuse=True)

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

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for batch in range(start_batch_id, self.num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images,  _,  manual_labels = train_val_data_iterator.get_next_batch("train")
                batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run( [self.optim,
                                                                          self.merged_summary_op,
                                                                          self.loss,
                                                                          self.neg_loglikelihood,
                                                                          self.KL_divergence],
                                                                         feed_dict={self.inputs: batch_images,
                                                                                    self.standard_normal: batch_z})
                # print(f"Epoch:{epoch} Batch:{batch}  loss={loss} nll={nll_loss} kl_loss={kl_loss}")
                self.counter += 1
                self.num_training_epochs_completed = epoch
                self.num_steps_completed = batch
                if self.exp_config.run_evaluation_during_training:
                    if np.mod(batch, self.exp_config.eval_interval) == self.exp_config.eval_interval - 1:
                        train_val_data_iterator.reset_counter("val")
                        self.evaluate(data_iterator=train_val_data_iterator,
                                      dataset_type="val")
                        self.writer.add_summary(summary_str, self.counter - 1)
                    else:
                        self.writer.add_summary(summary_str, self.counter - 1)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # save model
            print(f"Epoch:{epoch}   loss={loss} nll={nll_loss} kl_loss={kl_loss}")

            print("Saving check point", self.exp_config.TRAINED_MODELS_PATH)
            self.save(self.exp_config.TRAINED_MODELS_PATH, self.counter)
            train_val_data_iterator.reset_counter("train")
            if np.mod(epoch, self.exp_config.model_save_interval) == 0:
                self.save(self.exp_config.TRAINED_MODELS_PATH, self.counter)

    def evaluate(self, data_iterator, dataset_type, num_batches_train=0, save_images=True ):
        if num_batches_train == 0:
            num_batches_train = self.num_batches_train
        print(f"Running evaluation after epoch:{self.num_training_epochs_completed} and step:{self.num_steps_completed} ")
        start_eval_batch = 0
        reconstructed_images = []
        num_eval_batches = data_iterator.get_num_samples(dataset_type) // self.exp_config.BATCH_SIZE
        manifold_w = 4
        tot_num_samples = min(self.sample_num, self.exp_config.BATCH_SIZE)
        manifold_h = tot_num_samples // manifold_w
        mu = None
        sigma = None
        z = None
        for batch_no in range(start_eval_batch, num_eval_batches):
            batch_eval_images, batch_labels, manual_labels = data_iterator.get_next_batch(dataset_type)
            if batch_eval_images.shape[0] < self.exp_config.BATCH_SIZE:
                data_iterator.reset_counter(dataset_type)
                break
            batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

            reconstructed_image, summary, mu_for_batch, sigma_for_batch, z_for_batch = self.sess.run([self.out,
                                                                                                     self.merged_summary_op,
                                                                                                     self.mu,
                                                                                                     self.sigma,
                                                                                                     self.z
                                                                                                      ],
                                                                                                     feed_dict={
                                                                                                         self.inputs: batch_eval_images,
                                                                                                         self.standard_normal: batch_z
                                                                                                     }
                                                                                                     )
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

        # encoded_df = pd.DataFrame(np.transpose(np.vstack([labels, labels_predicted])),
        #                           columns=["label", "label_predicted"])
        if self.exp_config.return_latent_vector:
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(self.exp_config.Z_DIM)
            encoded_df = pd.DataFrame(mu, columns=mean_col_names)
            #print(encoded_df.shape, mu.shape)
            #
            # # encoded_df[mean_col_names] = mu
            # for i, mean_col_name in enumerate(mean_col_names):
            #     print(encoded_df.shape, mu.shape, mu[:, i].shape)
            #     encoded_df[mean_col_name] = mu[:, i]

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

    def encode_and_get_features(self, images):
        mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = self.sess.run([self.mu,
                                                                               self.sigma,
                                                                               self.z,
                                                                               self.dense2_en,
                                                                               self.reshaped_en,
                                                                               self.conv2,
                                                                               self.conv1],
                                                                              feed_dict={self.inputs: images})

        return mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en

    def decode_and_get_features(self, z):
        batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

        images, dense1_de, dense2_de, reshaped_de, deconv1_de = self.sess.run([self.out,
                                                                               self.dense1_de,
                                                                               self.dense2_de,
                                                                               self.reshaped_de,
                                                                               self.deconv1_de
                                                                               ],
                                                                              feed_dict={self.z: z,
                                                                                         self.standard_normal: batch_z
                                                                                         })

        return images, dense1_de, dense2_de, reshaped_de, deconv1_de

    def decode(self, z):
        images = self.sess.run(self.out, feed_dict={self.z: z})
        return images

    def decode_l3(self, z):
        images = self.sess.run(self.out, feed_dict={self.dense2_en: z})
        return images

    def decode_layer1(self, z):
        batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)
        dense1_de = self.sess.run(self.dense1_de, feed_dict={self.z: z,
                                                             self.standard_normal: batch_z
                                                             })
        return dense1_de

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
