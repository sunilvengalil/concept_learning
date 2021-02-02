# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

import pandas as pd

from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao
from clearn.models.model import Model
from clearn.utils import prior_factory as prior
from clearn.utils.utils import save_image, save_single_image
from clearn.utils.dir_utils import get_eval_result_dir

import tensorflow as tf
from clearn.utils.tensorflow_wrappers import conv2d, linear, lrelu


class ClassifierModel(Model):
    _model_name = "ClassifierModel"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 num_units_in_layer=None,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 dao: IDao = MnistDao(),
                 write_predictions=True,
                 test_data_iterator=None
                 ):
        super().__init__(exp_config, sess, epoch)
        self.dao = dao
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
        self.counter, self.start_batch_id, self.start_epoch = self._initialize(train_val_data_iterator,
                                                                               read_from_existing_checkpoint,
                                                                               check_point_epochs)

    def _set_model_parameters(self):
        pass

#   Gaussian Encoder
    def _encoder(self, x, reuse=False):
        # Encoder models the probability  P(z/X)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        w = dict()
        b = dict()
        with tf.variable_scope("encoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                self.conv1 = lrelu(conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1'))
                self.conv2 = lrelu((conv2d(self.conv1, self.n[1], 3, 3, 2, 2, name='en_conv2')))
                self.reshaped_en = tf.reshape(self.conv2, [self.exp_config.BATCH_SIZE, -1])
                self.dense2_en = lrelu(linear(self.reshaped_en, self.n[2], scope='en_fc3'))
            elif self.exp_config.activation_hidden_layer == "LINEAR":
                self.conv1 = conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1')
                self.conv2 = (conv2d(self.conv1, self.n[1], 3, 3, 2, 2, name='en_conv2'))
                self.reshaped_en = tf.reshape(self.conv2, [self.exp_config.BATCH_SIZE, -1])
                self.dense2_en = linear(self.reshaped_en, self.n[2], scope='en_fc3')
            else:
                raise Exception(f"Activation {self.exp_config.activation} not supported")

            # with tf.control_dependencies([net_before_gauss]):
            gaussian_params, w["en_fc4"], b["en_fc4"] = linear(self.dense2_en, 2 * self.exp_config.Z_DIM,
                                                               scope='en_fc4',
                                                               with_w=True)

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.exp_config.Z_DIM]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.exp_config.Z_DIM:])
        return mean, stddev

    def _build_model(self):
        # some parameters
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        self.standard_normal = tf.placeholder(tf.float32, [bs, self.exp_config.Z_DIM], name='z')

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

        self.loss = self.exp_config.reconstruction_weight * self.neg_loglikelihood +\
                    self.exp_config.beta * self.KL_divergence +\
                    self.exp_config.supervise_weight * self.supervised_loss
        # self.loss = -evidence_lower_bound + self.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.exp_config.learning_rate, beta1=self.beta_adam) \
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

    def get_trainable_vars(self):
        return tf.trainable_variables()

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
                batch_images,  _,  manual_labels = train_val_data_iterator.get_next_batch("train")
                batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss, supervised_loss = self.sess.run(
                    [self.optim, self.merged_summary_op,
                     self.loss, self.neg_loglikelihood,
                     self.KL_divergence, self.supervised_loss],
                    feed_dict={self.inputs: batch_images,
                               self.labels: manual_labels[:, :10],
                               self.is_manual_annotated: manual_labels[:, 10],
                               self.standard_normal: batch_z})

                counter += 1
                if self.exp_config.run_evaluation_during_training:
                    if np.mod(idx, self.exp_config.eval_interval) == self.exp_config.eval_interval - 1:
                        self.evaluate(epoch, idx + 1, counter - 1, val_data_iterator=train_val_data_iterator)
                        self.writer.add_summary(summary_str, counter-1)
                    else:
                        self.writer.add_summary(summary_str, counter-1)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            print("Saving check point", self.exp_config.TRAINED_MODELS_PATH)
            self.save(self.exp_config.TRAINED_MODELS_PATH, counter)
            train_val_data_iterator.reset_counter("train")
            if np.mod(epoch, self.exp_config.model_save_interval) == 0:
                self.save(self.exp_config.TRAINED_MODELS_PATH,counter)

    def evaluate(self, epoch, step, counter, val_data_iterator):
        print("Running evaluation after epoch:{:02d} and step:{:04d} ".format(epoch, step))
        # evaluate reconstruction loss
        start_eval_batch = 0
        reconstructed_images = []
        num_eval_batches = val_data_iterator.get_num_samples("val") // self.exp_config.BATCH_SIZE
        manifold_w = 4
        tot_num_samples = min(self.sample_num, self.exp_config.BATCH_SIZE)
        manifold_h = tot_num_samples // manifold_w

        for _idx in range(start_eval_batch, num_eval_batches):
            batch_eval_images, batch_eval_labels, manual_labels = val_data_iterator.get_next_batch("val")
            integer_label = np.asarray([np.where(r == 1)[0][0] for r in batch_eval_labels]).reshape([64, 1])
            batch_eval_labels = np.concatenate([batch_eval_labels, integer_label], axis=1)
            columns = [str(i) for i in range(10)]
            columns.append("label")
            pd.DataFrame(batch_eval_labels,
                         columns=columns)\
                .to_csv(self.exp_config.PREDICTION_RESULTS_PATH + "label_test_{:02d}.csv".format(_idx),
                        index=False)

            batch_z = prior.gaussian(self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)
            reconstructed_image, summary = self.sess.run([self.out,
                                                          self.merged_summary_op],
                                                         feed_dict={self.inputs: batch_eval_images,
                                                                    self.labels: manual_labels[:, :10],
                                                                    self.is_manual_annotated: manual_labels[:, 10],
                                                                    self.standard_normal: batch_z})
            training_batch = epoch * 935 + step
            save_single_image(reconstructed_image,
                              self.exp_config.reconstructed_images_path,
                              epoch, step, training_batch,
                              _idx, self.exp_config.BATCH_SIZE)

            self.writer_v.add_summary(summary, counter)
            reconstructed_images.append(reconstructed_image[:manifold_h * manifold_w, :, :, :])
        print(f"epoch:{epoch} step:{step}")
        reconstructed_dir = get_eval_result_dir(self.exp_config.PREDICTION_RESULTS_PATH, epoch, step)
        print(reconstructed_dir)

        for _idx in range(start_eval_batch, num_eval_batches):
            file = "im_" + str(_idx) + ".png"
            save_image(reconstructed_images[_idx], [manifold_h, manifold_w], reconstructed_dir + file)

        val_data_iterator.reset_counter("val")

        print("Evaluation completed")

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self._model_name_, self.exp_config.dataset_name,
            self.exp_config.BATCH_SIZE, self.exp_config.Z_DIM)


    def encode(self, images):
        mu, sigma, z, y_pred = self.sess.run([self.mu, self.sigma, self.z, self.y_pred],
                                             feed_dict={self.inputs: images})
        return mu, sigma, z, y_pred

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

    def classify(self, images):
        logits = self.sess.run([self.y_pred],
                               feed_dict={self.inputs: images})

        return logits
