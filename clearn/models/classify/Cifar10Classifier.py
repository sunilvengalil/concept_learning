import tensorflow as tf

from clearn.dao.idao import IDao
from clearn.dao.mnist import MnistDao
from clearn.models.architectures import cnn_4_layer
from clearn.utils.tensorflow_wrappers import conv2d, linear, deconv2d, lrelu
from clearn.models.classify.supervised_classifier import SupervisedClassifierModel


class Cifar10Classifier(SupervisedClassifierModel):
    _model_name = "ClassifierModel"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 num_units_in_layer=None,
                 dao: IDao = MnistDao(),
                 test_data_iterator=None
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         num_units_in_layer,
                         dao=dao,
                         test_data_iterator=test_data_iterator
                         )
        self.strides = [2, 2, 2, 2]

    def _encoder(self, x, reuse=False):
        return cnn_4_layer(self, x, self.exp_config.Z_DIM, reuse)

    def _build_model(self):
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
        self.strides = [2, 2]

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # random vectors with  multi-variate gaussian distribution
        # 0 mean and covariance matrix as Identity
        self.standard_normal = tf.compat.v1.placeholder(tf.float32, [bs, self.exp_config.Z_DIM], name='z')

        # Whether the sample was manually annotated.
        self.is_manual_annotated = tf.compat.v1.placeholder(tf.float32, [bs], name="is_manual_annotated")
        self.labels = tf.compat.v1.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        """ Loss Function """
        # encoding
        self.z = self._encoder(self.inputs, reuse=False)

        # supervised loss for labelled samples

        self.y_pred = linear(self.z, self.dao.num_classes)

        self.supervised_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                                         logits=self.y_pred
                                                                         )
        self.loss = self.exp_config.supervise_weight * self.supervised_loss

        """ Training """
        # optimizers
        t_vars = tf.compat.v1.trainable_variables()
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.optim = tf.compat.v1.train.AdamOptimizer(0.001) \
                .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        """ Summary """
        tf.compat.v1.summary.scalar("Supervised Loss", self.supervised_loss)
        tf.compat.v1.summary.scalar("Total Loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.compat.v1.summary.merge_all()
