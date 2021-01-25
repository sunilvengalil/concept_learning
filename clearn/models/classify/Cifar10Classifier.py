import tensorflow as tf

from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao
from clearn.utils.tensorflow_wrappers import conv2d, linear, deconv2d, lrelu
from clearn.models.classify.supervised_classifier import  SupervisedClassifierModel


class Cifar10Classifier(SupervisedClassifierModel):
    _model_name = "ClassifierModel"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 batch_size,
                 z_dim,
                 dataset_name,
                 beta=5,
                 num_units_in_layer=None,
                 log_dir=None,
                 checkpoint_dir=None,
                 result_dir=None,
                 train_val_data_iterator=None,
                 read_from_existing_checkpoint=True,
                 check_point_epochs=None,
                 supervise_weight=0,
                 reconstruction_weight=1,
                 reconstructed_image_dir=None,
                 dao: IDao = MnistDao(),
                 write_predictions=True
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         batch_size,
                         z_dim,
                         dataset_name,
                         beta,
                         num_units_in_layer,
                         log_dir,
                         checkpoint_dir,
                         result_dir,
                         train_val_data_iterator,
                         read_from_existing_checkpoint,
                         check_point_epochs,
                         supervise_weight,
                         reconstruction_weight,
                         reconstructed_image_dir,
                         dao,
                         write_predictions)
        self.strides = [2, 2, 2, 2]

    def _encoder(self, x, reuse=False):
        # Encoder models the probability  P(z/X)
        #pytorch code
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(True),
        #     View((-1, 1024*2*2)),                                 # B, 1024*4*4
        # )

        with tf.compat.v1.variable_scope("encoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                conv1 = conv2d(x, self.n[0], 3, 3, self.strides[0], self.strides[0], name='en_conv1')
                conv1 = tf.compat.v1.layers.batch_normalization(conv1)
                self.conv1 =lrelu((conv1))

                conv2 = conv2d(self.conv1, self.n[1], 3, 3, self.strides[0], self.strides[0], name='en_conv2')
                conv2 = tf.compat.v1.layers.batch_normalization(conv2)
                self.conv2 =lrelu((conv2))

                conv3 = conv2d(self.conv2, self.n[2], 3, 3, self.strides[0], self.strides[0], name='en_conv3')
                conv3 = tf.compat.v1.layers.batch_normalization(conv3)
                self.conv3 = lrelu((conv3))

                conv4 = conv2d(self.conv3, self.n[3], 3, 3, self.strides[0], self.strides[0], name='en_conv4')
                conv4 = tf.compat.v1.layers.batch_normalization(conv4)
                self.conv4 = lrelu((conv4))

                reshaped = tf.reshape(self.conv4, [self.batch_size, -1])

                # self.dense2_en = lrelu(linear(self.reshaped_en, self.n[2], scope='en_fc3'))

            else:
                raise Exception(f"Activation {self.exp_config.activation} not implemented")

            z = linear(reshaped,
                       self.z_dim,
                       scope='en_fc1')
        return z

    # Bernoulli decoder
    def decoder(self, z, reuse=False):
        # py pytorch code
        # self.fc_mu = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        # self.fc_logvar = nn.Linear(1024 * 2 * 2, z_dim)  # B, z_dim
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 1024 * 4 * 4),  # B, 1024*8*8
        #     View((-1, 1024, 4, 4)),  # B, 1024,  8,  8
        #     nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
        # )

        # Models the probability P(X/z)
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S

        output_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
        layer_4_size = [self.batch_size,
                        output_shape[1] // self.strides[0],
                        output_shape[2] // self.strides[0],
                        self.n[0]]
        layer_3_size = [self.batch_size,
                        layer_4_size[1] // self.strides[1],
                        layer_4_size[2] // self.strides[1],
                        self.n[1]]
        layer_2_size = [self.batch_size,
                        layer_3_size[1] // self.strides[2],
                        layer_3_size[2] // self.strides[2],
                        self.n[2]]
        layer_1_size = [self.batch_size,
                        layer_2_size[1] // self.strides[3],
                        layer_2_size[2] // self.strides[3],
                        self.n[3]]

        with tf.variable_scope("decoder", reuse=reuse):
            if self.exp_config.activation_hidden_layer == "RELU":
                # TODO remove hard coding

                self.dense1_de = linear(z, 1024 * 4 * 4, scope='de_fc1')
                #self.dense2_de = lrelu((linear(self.dense1_de, layer_2_size)))
                self.reshaped_de = tf.reshape(self.dense2_de, layer_1_size)
                deconv1 = lrelu(deconv2d(self.reshaped_de,
                                                 layer_2_size,
                                                 3, 3, self.strides[1], self.strides[1], name='de_dc1'))
                self.deconv1 = lrelu(tf.compat.v1.layers.batch_normalization(deconv1))

                deconv2 = lrelu(deconv2d(self.self.deconv1,
                                                 layer_3_size,
                                                 3, 3, self.strides[2], self.strides[2], name='de_dc2'))
                self.deconv2 = lrelu(tf.compat.v1.layers.batch_normalization(deconv2))

                deconv3 = lrelu(deconv2d(self.deconv2,
                                                 layer_4_size,
                                                 3, 3, self.strides[3], self.strides[3], name='de_dc3'))
                self.deconv3 = lrelu(tf.compat.v1.layers.batch_normalization(deconv3))

                out = lrelu(deconv2d(self.deconv3,
                                                 output_shape,
                                                 3, 3, self.strides[4], self.strides[4], name='de_dc4'))
            else:
                raise Exception(f"Activation {self.exp_config.activation} not supported")
            return out

    def _build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        self.standard_normal = tf.compat.v1.placeholder(tf.float32, [bs, self.z_dim], name='z')

        # Whether the sample was manually annotated.
        self.is_manual_annotated = tf.compat.v1.placeholder(tf.float32, [bs], name="is_manual_annotated")
        self.labels = tf.compat.v1.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        """ Loss Function """
        # encoding
        self.z = self._encoder(self.inputs, reuse=False)

        # supervised loss for labelled samples

        self.y_pred = linear(self.z, self.dao.num_classes)

        self.supervised_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                                         logits=self.y_pred,
                                                                         weights=self.is_manual_annotated
                                                                         )
        self.loss = self.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.compat.v1.trainable_variables()
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        """ Summary """
        tf.compat.v1.summary.scalar("Supervised Loss", self.supervised_loss)
        tf.compat.v1.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.compat.v1.summary.merge_all()
