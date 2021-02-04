import tensorflow as tf
from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao
from clearn.models.architectures.custom.tensorflow_graphs import cnn_4_layer, deconv_4_layer
from clearn.models.vae import VAE


class Cifar10Vae(VAE):

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess,
                 epoch,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 dao: IDao = MnistDao()
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         train_val_data_iterator=train_val_data_iterator,
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

    def decoder(self, z, reuse=False):
        # Models the probability P(X/z)
        return deconv_4_layer(self, z, reuse)
