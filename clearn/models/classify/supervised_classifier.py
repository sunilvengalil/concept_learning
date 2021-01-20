# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.special import softmax

from clearn.config.common_path import get_encoded_csv_file
from clearn.utils.utils import get_latent_vector_column
from clearn.utils import prior_factory as prior


from clearn.models.classify.classifier import ClassifierModel
import tensorflow as tf
from clearn.utils.tensorflow_wrappers import conv2d, linear, deconv2d, lrelu


class SupervisedClassifierModel(ClassifierModel):
    _model_name = "ClassifierModel"
    def __init__(self, exp_config, sess, epoch, batch_size,
                 z_dim, dataset_name, beta=5,
                 num_units_in_layer=None,
                 log_dir=None,
                 checkpoint_dir=None,
                 result_dir=None,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 supervise_weight=0,
                 reconstruction_weight=1,
                 reconstructed_image_dir=None
                 ):
        self.sess = sess
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_val_samples = 128
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.reconstructed_image_dir = reconstructed_image_dir
        self.beta = beta
        self.supervise_weight = supervise_weight
        self.reconstruction_weight = reconstruction_weight
        self.exp_config = exp_config
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
                self.n = [64, 128, 32, z_dim ]
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
        w = dict()
        b= dict()
        with tf.variable_scope("encoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                self.conv1 = lrelu(conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1'))
                self.conv2 = lrelu((conv2d(self.conv1, self.n[1], 3, 3, 2, 2, name='en_conv2')))
                self.reshaped_en = tf.reshape(self.conv2, [self.batch_size, -1])
                self.dense2_en = lrelu(linear(self.reshaped_en, self.n[2], scope='en_fc3'))
            elif self.exp_config.activation_hidden_layer == "LINEAR":
                self.conv1 = conv2d(x, self.n[0], 3, 3, 2, 2, name='en_conv1')
                self.conv2 = (conv2d(self.conv1, self.n[1], 3, 3, 2, 2, name='en_conv2'))
                self.reshaped_en = tf.reshape(self.conv2, [self.batch_size, -1])
                self.dense2_en = linear(self.reshaped_en, self.n[2], scope='en_fc3')
            else:
                raise Exception(f"Activation {self.exp_config.activation} not supported")

            # with tf.control_dependencies([net_before_gauss]):
            z, w["en_fc4"], b["en_fc4"] = linear(self.dense2_en, 2 * self.z_dim,
                                                               scope='en_fc4',
                                                               with_w=True)

        return z

    # Bernoulli decoder
    def decoder(self, z, reuse=False):
        # Models the probability P(X/z)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                self.dense1_de = lrelu((linear(z, self.n[2], scope='de_fc1')))
                self.dense2_de = lrelu((linear(self.dense1_de, self.n[1] * 7 * 7)))
                self.reshaped_de = tf.reshape(self.dense2_de, [self.batch_size, 7, 7, self.n[1]])
                self.deconv1_de = lrelu(
                    deconv2d(self.reshaped_de, [self.batch_size, 14, 14, self.n[0]], 3, 3, 2, 2, name='de_dc3'))
                if self.exp_config.activation_output_layer == "SIGMOID":
                    out = tf.nn.sigmoid(deconv2d(self.deconv1_de, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
                elif self.exp_config.activation_output_layer == "LINEAR":
                    out = deconv2d(self.deconv1_de, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4')
            elif self.exp_config.activation_hidden_layer == "LINEAR":
                self.dense1_de = linear(z, self.n[2], scope='de_fc1')
                self.dense2_de = linear(self.dense1_de, self.n[1] * 7 * 7)
                self.reshaped_de = tf.reshape(self.dense2_de , [self.batch_size, 7, 7, self.n[1]])
                self.deconv1_de = deconv2d(self.reshaped_de, [self.batch_size, 14, 14, self.n[0]], 3, 3, 2, 2, name='de_dc3')
                if self.exp_config.activation_output_layer == "SIGMOID":
                    out = tf.nn.sigmoid(deconv2d(self.deconv1_de, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
                elif self.exp_config.activation_output_layer == "LINEAR":
                    out = deconv2d(self.deconv1_de, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4')
            else:
                raise Exception(f"Activation {self.exp_config.activation} not supported")
            # out = lrelu(deconv2d(deconv1, [self.batch_size, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            return out


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
        self.z = self._encoder(self.inputs, reuse=False)

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

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.loss = self.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """

        # for test
        """ Summary """
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
        num_batches_train = train_val_data_iterator.get_num_samples("train") // self.batch_size

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, num_batches_train):
                # first 10 elements of manual_labels is actual one hot encoded labels
                # and next value is confidence
                batch_images, _, manual_labels = train_val_data_iterator.get_next_batch("train")
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, supervised_loss = self.sess.run([self.optim,
                                                                       self.merged_summary_op,
                                                                       self.loss,
                                                                       self.supervised_loss],
                    feed_dict={self.inputs: batch_images,
                               self.labels: manual_labels[:, :10],
                               self.is_manual_annotated: manual_labels[:, 10],
                               self.standard_normal: batch_z})
                counter += 1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            print(f"Completed {epoch} epochs")
            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            self.evaluate(train_val_data_iterator, epoch,"train")
            self.evaluate(train_val_data_iterator, epoch,"val")

            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            start_batch_id = 0
            # save model
            self.save(self.checkpoint_dir, counter)
        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def evaluate(self, train_val_data_iterator, epoch, dataset_type="train", save_results=True):
        encoded_df = None
        while train_val_data_iterator.has_next(dataset_type):
            batch_images, batch_labels, _ = train_val_data_iterator.get_next_batch(dataset_type)
            if batch_images.shape[0] < self.exp_config.BATCH_SIZE:
                train_val_data_iterator.reset_counter(dataset_type)
                break

            z, z, z, y_pred = self.encode(batch_images)
            labels_predicted = softmax(y_pred)
            labels_predicted = np.argmax(labels_predicted, axis=1)
            z_dim = z.shape[1]
            mean_col_names, sigma_col_names, z_col_names, l3_col_names = get_latent_vector_column(z_dim)
            # TODO do this using numpy api
            labels = np.argmax(batch_labels,axis=1)
            # i = 0
            # labels = np.zeros()
            # for lbl in batch_labels:
            #     labels[i] = np.where(lbl == 1)[0][0]
            #     i += 1
            # print("labels_predicted shape",labels_predicted.shape)
            logit_column_names = ["logits_"+str(i) for i in range(y_pred.shape[1])]
            temp_df1 = pd.DataFrame(z, columns=z_col_names)
            temp_df2 = pd.DataFrame(y_pred,columns=logit_column_names)
            temp_df = pd.concat([temp_df1,temp_df2], axis=1)
            temp_df["label"] = labels
            temp_df["label_predicted"] = labels_predicted
            if encoded_df is not None:
                encoded_df = pd.concat([encoded_df, temp_df])
            else:
                encoded_df = temp_df
        print(self.exp_config.ANALYSIS_PATH)

        if save_results:
            output_csv_file = get_encoded_csv_file(self.exp_config, epoch, dataset_type)
            encoded_df.to_csv(os.path.join(self.exp_config.ANALYSIS_PATH, output_csv_file), index=False)
        return encoded_df

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

    def encode(self, images):
        z, y_pred = self.sess.run([ self.z, self.y_pred],
                                             feed_dict={self.inputs: images})
        return z, z, z, y_pred


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
        z, dense2_en, reshaped, conv2_en, conv1_en = self.sess.run([self.z,
                                                                       self.dense2_en,
                                                                       self.reshaped_en,
                                                                       self.conv2,
                                                                       self.conv1],
                                                                      feed_dict={self.inputs: images})

        return z, z, z, dense2_en, reshaped, conv2_en, conv1_en