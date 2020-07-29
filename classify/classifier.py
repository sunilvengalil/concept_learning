# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import numpy as np

import pandas as pd
from utils import prior_factory as prior
from utils.utils import save_image
from utils.dir_utils import get_eval_result_dir

import tensorflow as tf
from tensorflow_wrappers.layers import conv2d, linear, deconv2d, lrelu


class ClassifierModel(object):
    _model_name = "ClassifierModel"

    def __init__(self, sess, epoch, batch_size,
                 z_dim, dataset_name, beta=5,
                 num_units_in_layer=None,
                 log_dir=None,
                 checkpoint_dir=None,
                 result_dir=None,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 supervise_weight=0,
                 reconstruction_weight=1
                 ):
        self.sess = sess
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_val_samples = 128
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.beta = beta
        self.supervise_weight = supervise_weight
        self.reconstruction_weight = reconstruction_weight
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.label_dim = 10  # one hot encoding for 10 classes
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 1
            if num_units_in_layer is None or len(num_units_in_layer) == 0:
                self.n = [64, 128, 32, z_dim * 2]
            else:
                self.n = num_units_in_layer
            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved
            self.num_images_per_row = 4  # should be a factor of sample_num
            self.eval_interval = 300
            # self.num_eval_batches = 10
        else:
            raise NotImplementedError("Dataset {} not implemented".format(dataset_name))
        self.mu = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='mu')
        self.sigma = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='sigma')
        self.images = None
        self._build_model()
        # initialize all variables
        tf.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)
        self.max_to_keep = 20

        self.counter, self.start_batch_id, self.start_epoch = self._initialize(train_val_data_iterator,
                                                                               read_from_existing_checkpoint,
                                                                               check_point_epochs)

#   Gaussian Encoder
    def _encoder(self, x, reuse=False):
        # Encoder models the probability  P(z/X)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):
            self.conv1 = lrelu(conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1'))
            self.conv2 = lrelu((conv2d(self.conv1, self.n[1], 3, 3, 2, 2, name='en_conv2')))
            self.reshaped_en = tf.reshape(self.conv2, [self.batch_size, -1])
            self.dense2_en = lrelu(linear(self.reshaped_en, self.n[2], scope='en_fc3'))
            # net_before_gauss = tf.print('shape of net is ', tf.shape(net))

            # with tf.control_dependencies([net_before_gauss]):
            gaussian_params = linear(self.dense2_en, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])
        return mean, stddev

    # Bernoulli decoder
    def decoder(self, z, reuse=False):
        # Models the probability P(X/z)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            dense1 = lrelu((linear(z, self.n[2], scope='de_fc1')))
            dense2 = lrelu((linear(dense1, self.n[1] * 7 * 7)))
            reshaped = tf.reshape(dense2, [self.batch_size, 7, 7, self.n[1]])
            deconv1 = lrelu(
                deconv2d(reshaped, [self.batch_size, 14, 14, self.n[0]], 3, 3, 2, 2, name='de_dc3'))

            out = tf.nn.sigmoid(deconv2d(deconv1, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            # out = lrelu(deconv2d(deconv1, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            return out

    def inference(self):
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        self.images = self.decoder(z, reuse=True)

    def _build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        self.standard_normal = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        # Whether the sample was manually annotated.
        self.is_manual_annotated = tf.placeholder(tf.float32, [bs], name="is_manual_annotated")
        self.labels = tf.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        """ Loss Function """
        # encoding
        self.mu, self.sigma = self._encoder(self.inputs, reuse=False)

        # sampling by re-parameterization technique
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # supervised loss for labelled samples
        self.y_pred = linear(self.z, 10)
        self.supervised_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                               logits=self.y_pred,
                                                               weights=self.is_manual_annotated
                                                               )

        # decoding
        out = self.decoder(self.z, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) +
                                            (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])
        kl = 0.5 * tf.reduce_sum(tf.square(self.mu) +
                                 tf.square(self.sigma) -
                                 tf.log(1e-8 + tf.square(self.sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(kl)

        # evidence_lower_bound = -self.neg_loglikelihood - self.beta * self.KL_divergence

        self.loss = self.reconstruction_weight * self.neg_loglikelihood + \
                    self.beta * self.KL_divergence + \
                    self.supervise_weight * self.supervised_loss
        # self.loss = -evidence_lower_bound + self.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
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

        # counter, start_batch_id, start_epoch = self.initialize(train_val_data_iterator)
        counter = self.counter
        start_batch_id = self.start_batch_id
        start_epoch = self.start_epoch
        self.evaluate(start_epoch, start_batch_id, 0, val_data_iterator=train_val_data_iterator)
        num_batches_train = train_val_data_iterator.get_num_samples("train") // self.batch_size

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, num_batches_train):
                # first 10 elements of manual_labels is actual one hot endoded labels
                # and next value is confidence currently binary(0/1). TODO change this to continuous
                #
                batch_images, _, manual_labels = train_val_data_iterator.get_next_batch("train")
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss, supervised_loss = self.sess.run(
                    [self.optim, self.merged_summary_op,
                     self.loss, self.neg_loglikelihood,
                     self.KL_divergence, self.supervised_loss],
                    feed_dict={self.inputs: batch_images,
                               self.labels: manual_labels[:, :10],
                               self.is_manual_annotated: manual_labels[:, 10],
                               self.standard_normal: batch_z})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f, supervised_loss: %.4f"
                      % (epoch, idx, num_batches_train, time.time() - start_time, loss, nll_loss, kl_loss,
                         supervised_loss))
                counter += 1
                if np.mod(idx, self.eval_interval) == 0:
                    self.evaluate(epoch + 1, idx - 1, counter - 1, val_data_iterator=train_val_data_iterator)
                    self.writer.add_summary(summary_str, counter-1)
                else:
                    self.writer.add_summary(summary_str, counter-1)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            print("Saving check point", self.checkpoint_dir)
            self.save(self.checkpoint_dir, counter)
            train_val_data_iterator.reset_counter("train")

            # show temporal results
        # self.visualize_results()

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def _initialize(self, train_val_data_iterator=None,
                    restore_from_existing_checkpoint=True,
                    check_point_epochs=None):
        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=50)
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self._model_name,
                                            self.sess.graph)
        self.writer_v = tf.summary.FileWriter(self.log_dir + '/' + self._model_name + "_v",
                                              self.sess.graph)

        if train_val_data_iterator is not None:
            num_batches_train = train_val_data_iterator.get_num_samples("train") // self.batch_size

        if restore_from_existing_checkpoint:
            # restore check-point if it exits
            could_load, checkpoint_counter = self._load(self.checkpoint_dir,
                                                        check_point_epochs=check_point_epochs)
            if could_load:
                if train_val_data_iterator is not None:
                    start_epoch = int(checkpoint_counter / num_batches_train)
                    start_batch_id = checkpoint_counter - start_epoch * num_batches_train
                else:
                    start_epoch = -1
                    start_batch_id = -1
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                start_epoch = 0
                start_batch_id = 0
                counter = 1
                print(" [!] Load failed...")
        else:
            counter = 1
            start_epoch = 0
            start_batch_id = 0
        return counter, start_batch_id, start_epoch

    def evaluate(self, epoch, step, counter, val_data_iterator):
        print("Running evaluation after epoch:{:02d} and step:{:04d} ".format(epoch, step))
        # evaluate reconstruction loss
        start_eval_batch = 0
        reconstructed_images = []
        num_eval_batches = val_data_iterator.get_num_samples("val") // self.batch_size
        for _idx in range(start_eval_batch, num_eval_batches):
            batch_eval_images, batch_eval_labels, manual_labels = val_data_iterator.get_next_batch("val")
            integer_label = np.asarray([np.where(r == 1)[0][0] for r in batch_eval_labels]).reshape([64, 1])
            batch_eval_labels = np.concatenate([batch_eval_labels, integer_label], axis=1)
            columns = [str(i) for i in range(10)]
            columns.append("label")
            pd.DataFrame(batch_eval_labels,
                         columns=columns)\
                .to_csv(self.result_dir + "label_test_{:02d}.csv".format(_idx),
                        index=False)

            batch_z = prior.gaussian(self.batch_size, self.z_dim)
            reconstructed_image, summary = self.sess.run([self.out,
                                                          self.merged_summary_op],
                                                         feed_dict={self.inputs: batch_eval_images,
                                                                    self.labels: manual_labels[:, :10],
                                                                    self.is_manual_annotated: manual_labels[:, 10],
                                                                    self.standard_normal: batch_z})

            self.writer_v.add_summary(summary, counter)

            manifold_w = 4
            tot_num_samples = min(self.sample_num, self.batch_size)
            manifold_h = tot_num_samples // manifold_w
            reconstructed_images.append(reconstructed_image[:manifold_h * manifold_w, :, :, :])
        print(f"epoch:{epoch} step:{step}")
        reconstructed_dir = get_eval_result_dir(self.result_dir, epoch, step)
        print(reconstructed_dir)

        for _idx in range(start_eval_batch, num_eval_batches):
            file = "im_" + str(_idx) + ".png"
            save_image(reconstructed_images[_idx], [manifold_h, manifold_w], reconstructed_dir + file)
        val_data_iterator.reset_counter("val")

        print("Evaluation completed")

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self._model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self._model_name + '.model'), global_step=step)

    def _load(self, checkpoint_dir, check_point_epochs=None):
        import re
        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=20)

        print(" [*] Reading checkpoints...")
        checkpoint_dir = checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if check_point_epochs is not None:
                ckpt_name = check_point_epochs
            print("ckpt_name", ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def generate_image(self, mu, sigma):
        self.inference()
        # original_samples = self.sess.run(self.images, feed_dict={self.mu: mu,self.sigma:sigma})
        original_samples = self.sess.run(self.images, feed_dict={self.mu: mu, self.sigma: sigma})
        return original_samples

    def encode(self, images):
        mu, sigma, z = self.sess.run([self.mu, self.sigma, self.z],
                                     feed_dict={self.inputs: images})
        return mu, sigma, z

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

    def decode(self, z):
        images = self.sess.run(self.out, feed_dict={self.z: z})
        return images

    def set_result_directories(self, log_dir, checkpoint_dir, result_dir):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir

    def load_from_checkpoint(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # restore check-point if it exits
        could_load, checkpoint_counter = self._load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        return checkpoint_counter
