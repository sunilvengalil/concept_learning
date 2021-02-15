import tensorflow as tf
from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.models.architectures.custom.tensorflow_graphs import cnn_4_layer, deconv_4_layer
from clearn.models.vae import VAE


class Cifar10Vae(VAE):

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 dao: IDao,
                 train_val_data_iterator=None,
                 test_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         train_val_data_iterator=train_val_data_iterator,
                         test_data_iterator=test_data_iterator,
                         read_from_existing_checkpoint=read_from_existing_checkpoint,
                         check_point_epochs=check_point_epochs,
                         dao=dao)

    def _encoder(self, x, reuse=False):
        gaussian_params = cnn_4_layer(self, x, 2 * self.exp_config.Z_DIM, reuse)
        # The mean parameter is unconstrained
        mean = gaussian_params[:, :self.exp_config.Z_DIM]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.exp_config.Z_DIM:])

        return mean, stddev

    def _decoder(self, z, reuse=False):
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
