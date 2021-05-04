from typing import Tuple

import tensorflow as tf
from clearn.utils.tensorflow_wrappers.layers import max_pool_2d, avgpool, flatten

from clearn.utils.tensorflow_wrappers import conv2d, lrelu, linear, deconv2d, drop_out


def cnn_n_layer(model, x, num_out_units, reuse=False):
    # Encoder models the probability  P(z/X)
    w = dict()
    b = dict()
    n_units = model.exp_config.num_units
    layer_num = 0
    with tf.compat.v1.variable_scope("encoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            model.final_conv = lrelu(conv2d(x, n_units[layer_num], 3, 3, 2, 2, name='en_conv1'))
            layer_num += 1
            if len(n_units) > 2:
                model.final_conv = lrelu((conv2d(model.final_conv, n_units[layer_num], 3, 3, 2, 2, name='en_conv2')))

            model.reshaped_en = tf.reshape(model.final_conv, [model.exp_config.BATCH_SIZE, -1])
            model.dense2_en = lrelu(linear(model.reshaped_en, n_units[layer_num], scope='en_fc2'))
        elif model.exp_config.activation_hidden_layer == "LINEAR":
            model.final_conv = conv2d(x, n_units[layer_num], 3, 3, 2, 2, name='en_conv1')
            layer_num += 1
            if len(n_units) > 2:
                model.final_conv = conv2d(model.final_conv, n_units[1], 3, 3, 2, 2, name='en_conv2')
            model.reshaped_en = tf.reshape(model.final_conv, [model.exp_config.BATCH_SIZE, -1])
            model.dense2_en = linear(model.reshaped_en, n_units[layer_num], scope='en_fc2')
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        # with tf.control_dependencies([net_before_gauss]):
        z, w["en_fc3"], b["en_fc3"] = linear(model.dense2_en, num_out_units,
                                             scope='en_fc3',
                                             with_w=True)
        return z

def fcnn_n_layer(model, x, num_out_units, reuse=False):
    # Encoder models the probability  P(z/X)
    n_units = model.exp_config.num_units
    layer_num = 0
    strides = model.exp_config.strides
    model.encoder_dict ={}
    with tf.compat.v1.variable_scope("encoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            x = add_zero_padding(x, model.padding_added_row[layer_num], model.padding_added_col[layer_num] )

            model.encoder_dict[f"layer_{layer_num}"] = lrelu(conv2d(x,
                                                                    n_units[layer_num],
                                                                    3, 3,
                                                                    strides[layer_num],
                                                                    strides[layer_num],
                                                                    name=f"layer_{layer_num}")
                                                             )
            for layer_num in range(1, len(n_units)):
                model.encoder_dict[f"layer_{layer_num - 1}"] = add_zero_padding(model.encoder_dict[f"layer_{layer_num-1}"],
                                                                                model.padding_added_row[layer_num],
                                                                                model.padding_added_col[layer_num]
                                                                                )
                model.encoder_dict[f"layer_{layer_num}"] = lrelu((conv2d(model.encoder_dict[f"layer_{layer_num - 1}"],
                                                                           n_units[layer_num],
                                                                           3, 3,
                                                                           strides[layer_num],
                                                                           strides[layer_num],
                                                                           name=f"layer_{layer_num}")))
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        z = lrelu((conv2d(model.encoder_dict[f"layer_{len(n_units) - 1}"],
                                    2,
                                    3, 3,
                                    strides[len(n_units)],
                                    strides[len(n_units)],
                                    name='out')))

        z = tf.reshape(z, [model.exp_config.BATCH_SIZE, -1])
        return z


def add_zero_padding(x:tf.Tensor, row_padding:Tuple[int], col_padding:Tuple[int]):
    if row_padding[0] != 0 or row_padding[1] != 0:
        x = tf.compat.v1.pad(x, [[0, 0], row_padding, [0, 0], [0, 0]])
    if col_padding[0] != 0 or col_padding[1] != 0:
        x = tf.compat.v1.pad(x, [[0, 0], [0, 0], col_padding, [0, 0]])
    return x


def remove_padding(x, row_padding, col_padding):
    x = x[:, row_padding[0]: x.shape[1] - row_padding[1], col_padding[0]: x.shape[2] - col_padding[1], :]
    return x


def fully_deconv_n_layer(model, z, reuse=False):
    n_units = model.exp_config.num_units
    h, w = model.dao.image_shape[0], model.dao.image_shape[1]
    strides = model.exp_config.strides
    re_scale_factor = get_rescale_factor_fcnn(strides)
    image_sizes = model.image_sizes

    model.decoder_dict ={}
    with tf.compat.v1.variable_scope("decoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            layer_num = 0
            stride = strides[len(n_units)]
            model.reshaped_de = tf.reshape(z,
                                           [model.exp_config.BATCH_SIZE,
                                            image_sizes[len(n_units)][0],
                                            image_sizes[len(n_units)][1],
                                            1
                                            ]
                                           )

            re_scale_factor = re_scale_factor // stride
            de_convolved = lrelu(deconv2d(model.reshaped_de,
                                                                        [model.exp_config.BATCH_SIZE,
                                                                         image_sizes[len(n_units)][0],
                                                                         image_sizes[len(n_units)][0],
                                                                         n_units[len(n_units) - 1]],
                                                                        3, 3,
                                                                        stride,
                                                                        stride,
                                                                        name=f"de_conv_{layer_num}"))
            # padding_removed = remove_padding(de_convolved, model.padding_added_row[layer_num], model.padding_added_col[layer_num])
            model.decoder_dict[f"de_conv_{layer_num}"] = de_convolved
            for layer_num in range(1, len(n_units)):
                re_scale_factor = re_scale_factor// strides[len(n_units) - layer_num]
                de_convolved = lrelu(deconv2d(model.decoder_dict[f"de_conv_{layer_num - 1}"],
                                                                            [model.exp_config.BATCH_SIZE,
                                                                            image_sizes[len(n_units) - layer_num][0],
                                                                            image_sizes[len(n_units) - layer_num][0],
                                                                            n_units[len(n_units) - layer_num - 1]],
                                                                            3, 3,
                                                                            strides[len(n_units) - layer_num],
                                                                            strides[len(n_units) - layer_num],
                                                                            name=f"de_conv_{layer_num}"
                                                                            )
                                                                   )
                # padding_removed = remove_padding(de_convolved, model.padding_added_row[layer_num], model.padding_added_col[layer_num])
                model.decoder_dict[f"de_conv_{layer_num}"] = de_convolved
            if model.exp_config.activation_output_layer == "SIGMOID":
                out = tf.nn.sigmoid(
                    deconv2d(model.decoder_dict[f"de_conv_{len(n_units) - 1}"], [model.exp_config.BATCH_SIZE, h, w, 1], 3, 3, strides[0], strides[0], name='de_out'))
            elif model.exp_config.activation_output_layer == "LINEAR":
                out = deconv2d(model.decoder_dict[f"de_conv_{len(n_units) - 1}"], [model.exp_config.BATCH_SIZE, h, w, 1], 3, 3, strides[0], strides[0], name='de_out')
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        return out

def deconv_n_layer(model, z, reuse=False):
    n_units = model.exp_config.num_units
    h, w = model.dao.image_shape[0], model.dao.image_shape[1]
    re_scale_factor = get_rescale_factor(n_units)

    with tf.compat.v1.variable_scope("decoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            layer_num = 1
            model.dense1_de = lrelu((linear(z, n_units[len(n_units) - layer_num], scope='de_fc1')))
            layer_num += 1
            model.dense2_de = lrelu((linear(model.dense1_de, n_units[ len(n_units) - layer_num ] * h//re_scale_factor * w//re_scale_factor)))
            model.reshaped_de = tf.reshape(model.dense2_de, [model.exp_config.BATCH_SIZE, h//re_scale_factor, w//re_scale_factor, n_units[len(n_units) - layer_num]])

            if len(n_units) > 2:
                layer_num += 1
                re_scale_factor = re_scale_factor//2
                model.deconv1_de = lrelu(
                    deconv2d(model.reshaped_de, [model.exp_config.BATCH_SIZE, h//re_scale_factor, w//re_scale_factor, n_units[len(n_units) - layer_num]], 3, 3, 2, 2, name='de_dc3'))
            if model.exp_config.activation_output_layer == "SIGMOID":
                out = tf.nn.sigmoid(
                    deconv2d(model.reshaped_de, [model.exp_config.BATCH_SIZE, h, w, 1], 3, 3, 2, 2, name='de_dc3'))
            elif model.exp_config.activation_output_layer == "LINEAR":
                out = deconv2d(model.reshaped_de, [model.exp_config.BATCH_SIZE, h, w, 1], 3, 3, 2, 2, name='de_dc3')
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        # out = lrelu(deconv2d(deconv1, [self.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
        return out


def get_rescale_factor(n_units):
    if len(n_units) > 2:
        re_scale_factor = 4
    else:
        re_scale_factor = 2
    return re_scale_factor

def get_rescale_factor_fcnn(strides):
    rescale_factor = 1
    for stride in strides:
        rescale_factor = rescale_factor * stride
    return rescale_factor


def cnn_3_layer(model, x, num_out_units, reuse=False):
    # Encoder models the probability  P(z/X)
    w = dict()
    b = dict()
    n_units = model.exp_config.num_units
    n = len(n_units)
    with tf.compat.v1.variable_scope("encoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            model.conv1 = lrelu(conv2d(x, n_units[0], 3, 3, 2, 2, name='en_conv1'))
            model.conv2 = lrelu((conv2d(model.conv1, n_units[1], 3, 3, 2, 2, name='en_conv2')))
            model.reshaped_en = tf.reshape(model.conv2, [model.exp_config.BATCH_SIZE, -1])
            model.dense2_en = lrelu(linear(model.reshaped_en, n_units[2], scope='en_fc3'))
        elif model.exp_config.activation_hidden_layer == "LINEAR":
            model.conv1 = conv2d(x, n_units[0], 3, 3, 2, 2, name='en_conv1')
            model.conv2 = (conv2d(model.conv1, n_units[1], 3, 3, 2, 2, name='en_conv2'))
            model.reshaped_en = tf.reshape(model.conv2, [model.exp_config.BATCH_SIZE, -1])
            model.dense2_en = linear(model.reshaped_en, n_units[2], scope='en_fc3')
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        # with tf.control_dependencies([net_before_gauss]):
        z, w["en_fc4"], b["en_fc4"] = linear(model.dense2_en, num_out_units,
                                             scope='en_fc4',
                                             with_w=True)
        return z


def deconv_3_layer(model, z, reuse=False):
    # Models the probability P(X/z)
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    n = model.exp_config.num_units
    with tf.compat.v1.variable_scope("decoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            model.dense1_de = lrelu((linear(z, n[2], scope='de_fc1')))
            model.dense2_de = lrelu((linear(model.dense1_de, n[1] * 7 * 7)))
            model.reshaped_de = tf.reshape(model.dense2_de, [model.exp_config.BATCH_SIZE, 7, 7, n[1]])
            model.deconv1_de = lrelu(
                deconv2d(model.reshaped_de, [model.exp_config.BATCH_SIZE, 14, 14, n[0]], 3, 3, 2, 2, name='de_dc3'))
            if model.exp_config.activation_output_layer == "SIGMOID":
                out = tf.nn.sigmoid(
                    deconv2d(model.deconv1_de, [model.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            elif model.exp_config.activation_output_layer == "LINEAR":
                out = deconv2d(model.deconv1_de, [model.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4')
        elif model.exp_config.activation_hidden_layer == "LINEAR":
            model.dense1_de = linear(z, n[2], scope='de_fc1')
            model.dense2_de = linear(model.dense1_de, n[1] * 7 * 7)
            model.reshaped_de = tf.reshape(model.dense2_de, [model.exp_config.BATCH_SIZE, 7, 7, n[1]])
            model.deconv1_de = deconv2d(model.reshaped_de, [model.exp_config.BATCH_SIZE, 14, 14, n[0]], 3, 3, 2, 2,
                                        name='de_dc3')
            if model.exp_config.activation_output_layer == "SIGMOID":
                out = tf.nn.sigmoid(
                    deconv2d(model.deconv1_de, [model.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
            elif model.exp_config.activation_output_layer == "LINEAR":
                out = deconv2d(model.deconv1_de, [model.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4')
        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not supported")
        # out = lrelu(deconv2d(deconv1, [self.exp_config.BATCH_SIZE, 28, 28, 1], 3, 3, 2, 2, name='de_dc4'))
        return out


def cnn_4_layer(model, x, num_out_units, reuse=False):
    # Encoder models the probability  P(z/X)
    # pytorch code
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
    n = model.exp_config.num_units
    with tf.compat.v1.variable_scope("encoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":
            conv1 = conv2d(x, n[0], 3, 3, 1, 1, name='en_conv1')
            conv1 = tf.compat.v1.layers.batch_normalization(conv1)
            conv1 = lrelu(conv1, 0.0)
            model.conv1 = max_pool_2d(conv1,kernel_size=2, strides=2)
            model.conv1 = drop_out(model.conv1, 0.3)

            conv2 = conv2d(model.conv1, n[1], 3, 3, 1, 1, name='en_conv2')
            conv2 = tf.compat.v1.layers.batch_normalization(conv2)
            conv2 = lrelu(conv2, 0.0)
            model.conv2 = max_pool_2d(conv2,kernel_size=2, strides=2)
            model.conv2 = drop_out(model.conv2, 0.3)

            conv3 = conv2d(model.conv2, n[2], 3, 3, 1, 1, name='en_conv3')
            conv3 = tf.compat.v1.layers.batch_normalization(conv3)
            conv3 = lrelu(conv3, 0.0)
            model.conv3 = max_pool_2d(conv3,kernel_size=2, strides=2)

            conv4 = conv2d(model.conv3, n[3], 3, 3, 1, 1, name='en_conv4')
            conv4 = tf.compat.v1.layers.batch_normalization(conv4)
            conv4 = lrelu(conv4, 0.01)
            model.conv4 = max_pool_2d(conv4, kernel_size=2, strides=2)

            model.reshaped = tf.reshape(model.conv3, [model.exp_config.BATCH_SIZE, -1])

            # self.dense2_en = lrelu(linear(reshaped, self.n[4], scope='en_fc1'), 0.0)

        else:
            raise Exception(f"Activation {model.exp_config.activation_hidden_layer} not implemented")

        z = linear(model.reshaped, num_out_units, scope='en_fc2')
    return z


# Bernoulli decoder
def deconv_4_layer(model, z, reuse=False):
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
    n = model.exp_config.num_units
    output_shape = [model.exp_config.BATCH_SIZE, model.dao.image_shape[0], model.dao.image_shape[1],
                    model.dao.image_shape[2]]
    layer_4_size = [model.exp_config.BATCH_SIZE,
                    output_shape[1] // model.exp_config.strides[0],
                    output_shape[2] // model.exp_config.strides[0],
                    n[0]]
    layer_3_size = [model.exp_config.BATCH_SIZE,
                    layer_4_size[1] // model.exp_config.strides[1],
                    layer_4_size[2] // model.exp_config.strides[1],
                    n[1]]
    layer_2_size = [model.exp_config.BATCH_SIZE,
                    layer_3_size[1] // model.exp_config.strides[2],
                    layer_3_size[2] // model.exp_config.strides[2],
                    n[2]]
    layer_1_size = [model.exp_config.BATCH_SIZE,
                    layer_2_size[1] // model.exp_config.strides[3],
                    layer_2_size[2] // model.exp_config.strides[3],
                    n[3]]

    with tf.compat.v1.variable_scope("decoder", reuse=reuse):
        if model.exp_config.activation_hidden_layer == "RELU":

            model.dense1_de = lrelu(linear(z, layer_1_size[1] * layer_1_size[2] * layer_1_size[3], scope="de_fc1"), 0)
            # self.dense2_de = lrelu(linear(self.dense1_de, 1024 * 4 * 4, scope='de_fc2'))
            model.reshaped_de = tf.reshape(model.dense1_de, layer_1_size)
            deconv1 = lrelu(deconv2d(model.reshaped_de,
                                     layer_2_size,
                                     3, 3, model.exp_config.strides[1], model.exp_config.strides[1], name='de_dc1'), 0)
            model.deconv1 = lrelu(tf.compat.v1.layers.batch_normalization(deconv1))

            deconv2 = lrelu(deconv2d(model.deconv1,
                                     layer_3_size,
                                     3, 3, model.exp_config.strides[2], model.exp_config.strides[2], name='de_dc2'), 0)
            deconv2 = lrelu(tf.compat.v1.layers.batch_normalization(deconv2))
            model.deconv2 = drop_out(deconv2, 0.3)

            deconv3 = lrelu(deconv2d(model.deconv2,
                                     layer_4_size,
                                     3, 3, model.exp_config.strides[3], model.exp_config.strides[3], name='de_dc3'), 0)
            model.deconv3 = lrelu(tf.compat.v1.layers.batch_normalization(deconv3))
            model.deconv3 = drop_out(deconv3, 0.3)


            if model.exp_config.activation_output_layer == "SIGMOID":
                out = tf.nn.sigmoid(
                    deconv2d(model.deconv3, output_shape, 3, 3, model.exp_config.strides[4], model.exp_config.strides[4], name='de_dc4'))
            elif model.exp_config.activation_output_layer == "LINEAR":
                out = lrelu(deconv2d(model.deconv3,
                                     output_shape,
                                     3, 3, model.exp_config.strides[4], model.exp_config.strides[4], name='de_dc4'),
                    0
                    )

        else:
            raise Exception(f"Activation {model.exp_config.activation} not supported")
        return out
