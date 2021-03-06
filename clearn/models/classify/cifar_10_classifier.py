import tensorflow as tf
from clearn.dao.idao import IDao
from clearn.models.architectures.custom.tensorflow_graphs import cnn_4_layer
from clearn.models.architectures.vgg import cfgs, classify, classify_f
from clearn.models.classify.supervised_classifier import SupervisedClassifierModel
from clearn.utils.tensorflow_wrappers import linear


class Cifar10Classifier(SupervisedClassifierModel):
    _model_name_ = "Cifar10Classifier"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 dao: IDao,
                 num_units_in_layer=None,
                 test_data_iterator=None,
                 check_point_epochs=None
                 ):
        self.strides = [2, 2, 2, 2]
        super().__init__(exp_config=exp_config,
                         sess=sess,
                         epoch=epoch,
                         num_units_in_layer=num_units_in_layer,
                         dao=dao,
                         test_data_iterator=test_data_iterator,
                         check_point_epochs=check_point_epochs
                         )

    def _encoder(self, x, reuse=False):
        return cnn_4_layer(self, x, self.exp_config.Z_DIM, reuse)


class Cifar10F(SupervisedClassifierModel):
    _model_name_ = "CiFar10_F"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 dao: IDao,
                 num_units_in_layer=None,
                 test_data_iterator=None,
                 check_point_epochs=None
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         num_units_in_layer=num_units_in_layer,
                         dao=dao,
                         test_data_iterator=test_data_iterator,
                         check_point_epochs=check_point_epochs
                         )

    def _encoder(self, x, reuse=False):
        z = classify_f(self, x, cfgs["F"], apply_batch_norm=False)
        return z

    def _build_model(self):
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
        self.strides = [2, 2]

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.labels = tf.compat.v1.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        # encoding
        self.z = self._encoder(self.inputs, reuse=False)

        # supervised loss for labelled samples
        self.y_pred = linear(self.z, self.dao.num_classes)

        self.compute_and_optimize_loss()



class Vgg16(SupervisedClassifierModel):
    _model_name_ = "CiFar10_VGG_16"

    def __init__(self,
                 exp_config,
                 sess,
                 epoch,
                 dao: IDao,
                 num_units_in_layer=None,
                 test_data_iterator=None
                 ):
        super().__init__(exp_config,
                         sess,
                         epoch,
                         num_units_in_layer,
                         dao=dao,
                         test_data_iterator=test_data_iterator
                         )

    def _encoder(self, x, reuse=False):
        return classify(self, x, cfgs["D"], apply_batch_norm=True)

    def _build_model(self):
        image_dims = self.dao.image_shape
        bs = self.exp_config.BATCH_SIZE
        self.strides = [2, 2]

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.labels = tf.compat.v1.placeholder(tf.float32, [bs, self.label_dim], name='manual_label')

        """ Loss Function """
        # supervised loss for labelled samples
        self.y_pred = self._encoder(self.inputs, reuse=False)

        self.compute_and_optimize_loss()
