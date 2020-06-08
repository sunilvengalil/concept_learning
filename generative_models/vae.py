# -*- coding: utf-8 -*-
from __future__ import division

import csv
import os
import time
from copy import deepcopy
import numpy as np

import pandas as pd
from utils import prior_factory as prior
from utils.utils import save_image, get_eval_result_dir

import tensorflow as tf
from tensorflow_wrappers.layers import conv2d, linear, deconv2d, lrelu
beta = 5

class VAE(object):
    _model_name = "VAE"

    def __init__(self, sess, epoch, batch_size,
                 z_dim, dataset_name, num_units_in_layer=None,
                 log_dir=None,
                checkpoint_dir=None,
                result_dir=None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.infer_ = 'infer_10/'
        self.orinfer_ = 'orinfer_10/'
        self.input_ = 'input_10/'
        self.num_val_samples = 128
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
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
            self.num_eval_batches = 10
        else:
            raise NotImplementedError("Dataset {} not implemented".format(dataset_name))
        self.mu = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='mu')
        self.sigma = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='sigma')
        self.images = None


#   Gaussian Encoder
    def encoder(self, x, reuse=False):
        # Encoder models the probability  P(z/X)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):
            net = lrelu(conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1'))
            net = lrelu((conv2d(net, self.n[1], 3, 3, 2, 2, name='en_conv2')))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(linear(net, self.n[2], scope='en_fc3'))
            # net_before_gauss = tf.print('shape of net is ', tf.shape(net))

            # with tf.control_dependencies([net_before_gauss]):
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

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
            net = lrelu( (linear(z, self.n[2], scope='de_fc1')) )
            net = lrelu((linear(net, self.n[1] * 7 * 7)))
            net = tf.reshape(net, [self.batch_size, 7, 7, self.n[1]])
            net = lrelu(
                deconv2d(net, [self.batch_size, 14, 14, self.n[0]], 3, 3, 2, 2, name='de_dc3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            return out

    def inference(self):
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        self.images = self.decoder(z, reuse=True)

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.standard_normal = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """
        # encoding
        self.mu, self.sigma = self.encoder(self.inputs, reuse=False)

        # sampling by re-parameterization technique
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out = self.decoder(self.z, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1,
                                            [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - beta * self.KL_divergence

        self.loss = -ELBO

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """

        # for test
        self.fake_images = self.decoder(self.standard_normal, reuse=True)

        # self.images = self.decoder(z, is_training=False, reuse=True)

        #

        """ Summary """
        nll_sum = tf.summary.scalar("Negative Log Likelihood", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("K L Divergence", self.KL_divergence)
        loss_sum = tf.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self,train_val_data_iterator):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self._model_name, self.sess.graph)
        self.writer_v = tf.summary.FileWriter(self.log_dir + '/' + self._model_name+"_v", self.sess.graph)


        self.num_batches_train = train_val_data_iterator.get_num_samples_train() // self.batch_size
        self.num_eval_batches =  train_val_data_iterator.get_num_samples_val() // self.batch_size

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches_train)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches_train
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        self.evaluate(start_epoch, start_batch_id, 0, val_data_iterator=train_val_data_iterator)

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches_train):
                batch_images = train_val_data_iterator.get_next_batch_train()
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run(
                    [self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                    feed_dict={self.inputs: batch_images, self.standard_normal: batch_z})

                # print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                #       % (epoch, idx, self.num_batches_train, time.time() - start_time, loss, nll_loss, kl_loss))


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
            train_val_data_iterator.reset_train_couner()

            # show temporal results
        # self.visualize_results()

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def evaluate(self, epoch, step, counter, val_data_iterator ):
        print("Running evaluation after epoch:{:02d} and step:{:04d} ".format(epoch, step))
        # evaluate reconstruction loss
        start_eval_batch = 0
        reconstructed_images = []
        for _idx in range(start_eval_batch, self.num_eval_batches):
            batch_eval_images, batch_eval_labels = val_data_iterator.get_next_batch_val()
            integer_label = np.asarray([np.where(r == 1)[0][0] for r in batch_eval_labels]).reshape([64, 1])
            batch_eval_labels = np.concatenate([batch_eval_labels, integer_label], axis=1)
            columns = [str(i) for i in range(10)]
            columns.append("label")
            pd.DataFrame(batch_eval_labels,
                         columns=columns)\
                .to_csv(self.result_dir + "label_test_{:02d}.csv".format(_idx),
                        index=False)

            # self.inference()

            # mu, sigma = self.encode(batch_eval_mages)
            batch_z = prior.gaussian(self.batch_size, self.z_dim)
            reconstructed_image, summary = self.sess.run([self.out, self.merged_summary_op],
                                                        feed_dict={self.inputs: batch_eval_images,
                                                        self.standard_normal: batch_z})


            self.writer_v.add_summary(summary, counter)

            manifold_w = 4
            tot_num_samples = min(self.sample_num, self.batch_size)
            manifold_h = tot_num_samples // manifold_w
            reconstructed_images.append(reconstructed_image[:manifold_h * manifold_w, :, :, :])
        print("epoch:{} step:{}".format(epoch,step))
        reconstructed_dir = get_eval_result_dir(self.result_dir,epoch,
                                                     step, "reconstructed")
        print(reconstructed_dir)

        for _idx in range(start_eval_batch, self.num_eval_batches):
            file = "im_" + str(_idx) + ".png"
            save_image(reconstructed_images[_idx], [manifold_h, manifold_w], reconstructed_dir + file)
        val_data_iterator.reset_val_couner()

        print("Evaluation completed")

    def save_val_images(self):
        start_eval_batch = 0
        original_images = []
        for _idx in range(start_eval_batch, self.num_eval_batches):
            batch_eval_mages = self.data_test_X[_idx * self.batch_size:(_idx + 1) * self.batch_size]
            batch_eval_labels = deepcopy(self.data_test_Y[_idx * self.batch_size:(_idx + 1) * self.batch_size])
            integer_label = np.asarray([np.where(r == 1)[0][0] for r in batch_eval_labels]).reshape([64, 1])
            batch_eval_labels = np.concatenate([batch_eval_labels, integer_label], axis=1)
            columns = [str(i) for i in range(10)]
            columns.append("label")

            pd.DataFrame(batch_eval_labels, columns=columns).to_csv(
                "/Users/sunilkumar/gitprojects/tensorflow-generative-model-collections/results/label_test_" + str(
                    _idx) + ".csv")

            manifold_w = 4
            tot_num_samples = min(self.sample_num, self.batch_size)
            manifold_h = tot_num_samples // manifold_w
            original_images.append(batch_eval_mages[:manifold_h * manifold_w, :, :, :])
        reconstructed_dir = self.get_eval_result_dir(orig_or_reconstructed="original")

        for _idx in range(start_eval_batch, self.num_eval_batches):
            file = "im_" + str(_idx) + ".png"
            save_image(original_images[_idx], [manifold_h, manifold_w], reconstructed_dir + file)

    def feature_analysis(self, mu, sigma, batch_images, batch_labels):
        reconsructed_images = self.sess.run(self.images , feed_dict={self.mu: mu, self.sigma: sigma})

        mu_max = np.max(mu)

        mu_min = np.min(mu)

        f_z = open("z.csv", "w")
        writer = csv.writer(f_z)
        for i in range(mu.shape[0]):
            # writer.writerow( [batch_labels[i]].extend( mu[i,:].tolist() ))
            tolist = mu[i, :].tolist()
            tolist.append(np.where(batch_labels[i] == 1)[0][0])
            writer.writerow(tolist)

        for j in range(np.shape(mu)[1]):

            original_samples = self.sess.run(self.images, feed_dict={self.mu: mu, self.sigma: sigma})

            indexes_to_keep = [a for a in list(range(j))]

            # mu_ = mu[:, indexes_to_keep]

            # mu_feat = mu[0, :]

            mu_ = np.copy(mu)

            mu_[:, :] = 0

            mu_[:, indexes_to_keep] = mu[:, indexes_to_keep]
            # mu_[:, j+1] = mu_min

            samples = self.sess.run(self.images, feed_dict={self.mu: mu_, self.sigma: sigma})

            # samples = self.sess.run(self.images, feed_dict={self.z: z})

            for i in range(len(samples)):
                sample = samples[i] * 255
                ori_sample = original_samples[i] * 255
                img = batch_images[i] * 255

                # cv2.imwrite(self.infer_ + str(j) + '_' + str(i) + '.jpg', sample)
                #
                # cv2.imwrite(self.orinfer_ + str(j) + '_' + str(i) + '.jpg', ori_sample)
                #
                # cv2.imwrite(self.input_ + str(j) + '_' + str(i) + '.jpg', img)
        f_z.close()

    def visualize_results(self,batch_images,batch_labels):
        mu, sigma = self.sess.run([self.mu, self.sigma], feed_dict={self.inputs: batch_images})
        self.inference()
        self.feature_analysis(mu, sigma, batch_images, batch_labels)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self._model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self._model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        # saver to save model
        self.saver = tf.train.Saver()

        print(" [*] Reading checkpoints...")
        checkpoint_dir = checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def restore_from_checkpoint(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self._model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def generate_image(self, mu, sigma):
        self.inference()
        # original_samples = self.sess.run(self.images, feed_dict={self.mu: mu,self.sigma:sigma})
        original_samples = self.sess.run(self.images, feed_dict={self.mu: mu, self.sigma: sigma})
        return original_samples

    def encode(self, images):
        mu, sigma,z = self.sess.run([self.mu, self.sigma, self.z], feed_dict={self.inputs: images})
        return mu, sigma, z

    def decode(self, z):
        images = self.sess.run(self.out, feed_dict={self.z: z})
        return images


    def set_result_directories(self,log_dir,checkpoint_dir,result_dir):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir

    def load_from_checkpoint(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        return checkpoint_counter