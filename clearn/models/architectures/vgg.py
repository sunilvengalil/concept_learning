from typing import *
import tensorflow as tf

from clearn.utils.tensorflow_wrappers import conv2d, linear, relu, drop_out, batch_norm
from clearn.models.classify.supervised_classifier import SupervisedClassifierModel
from clearn.utils.tensorflow_wrappers.layers import max_pool_2d, avgpool, flatten

# TODO get the corresponding tensorflow check points
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [32, 32, "M", 64, "M", ]
}
def classifier(model: SupervisedClassifierModel,
               features: tf.Tensor,
               num_out_units: int):
    model.fc1 = linear(features, 512 * 7 * 7, 4096)
    model.fc1_after_activation = relu(model.fc1)
    model.fc1_after_drop_out = drop_out(model.fc1_after_activation, 0.2)

    model.fc2 = linear(model.fc1_after_drop_out, 4096, 4096)
    model.fc2_after_activation = relu(model.fc2)
    model.fc2_after_drop_out = drop_out(model.fc2_after_activation, 0.2)

    out = linear(model.fc2_after_drop_out, 4096, num_out_units)
    return out


def classify_f(model: SupervisedClassifierModel, x, cfg: List, apply_batch_norm: bool):
    model.features = get_features(model, x, cfg, apply_batch_norm)
    model.features_flattened = flatten(model.features)
    #model.features_dropped_out = drop_out(model.features_flattened, 0.2)
    # TODO apply weight constraint
    # model.fc1 = linear(512, kernel_constraint(max_norm(3)))
    model.fc1 = linear(model.features_flattened, 512, scope="fc1")
    model.fc1_after_relu = relu(model.fc1)
    #z = drop_out(model.fc1_after_relu, 0.2)

    return model.fc1_after_relu


def classify(model: SupervisedClassifierModel, x, cfg: List, apply_batch_norm: bool):
    model.features = get_features(model, x, cfg, apply_batch_norm)
    model.features_pooled = avgpool(model.features, [7, 7])
    model.features_flattened = flatten(model.features_pooled)
    model.out = classifier(model, model.features_flattened, model.dao.num_classes)
    return model.out


def get_features(model: SupervisedClassifierModel,
                 x: tf.Tensor,
                 cfg: List,
                 bn: bool
                 ):
    model.layers = [x]
    conv_layer_index = 1
    for v in cfg:
        if v == 'M':
            model.layers.append(max_pool_2d(model.layers[-1], kernel_size=2, strides=2))
        elif v == 'D':
            model.layers.append(drop_out(model.layers[-1], 0.2))
        else:
            model.layers.append(conv2d(model.layers[-1],
                                       output_dim=v,
                                       k_h=3,
                                       k_w=3,
                                       d_h=1,
                                       d_w=1,
                                       name=f"conv_{conv_layer_index}"
                                       )
                                )
            conv_layer_index += 1
            if bn:
                model.layers.append(batch_norm(model.layers[-1]))
            model.layers.append(relu(model.layers[-1]))

    return model.layers[-1]


def _vgg(model: SupervisedClassifierModel,
         x: tf.Tensor,
         cfg: str,
         apply_batch_norm: bool,
         pretrained: bool,
         **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model.out = classify(model, x, cfgs[cfg], apply_batch_norm)
    #  TODO implement weight initialization from pre-trained net
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def cnn_3layer(model: SupervisedClassifierModel,
               x: tf.Tensor,
               pretrained: bool,
               **kwargs):
    return _vgg(model, x, cfg="F", batch_norm=False, pretrained=pretrained, **kwargs)


def vgg16(model: SupervisedClassifierModel,
          x: tf.Tensor,
          pretrained: bool,
          **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        model: Model object
        x: Input tensor
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg(model, x, cfg="D", batch_norm=False, pretrained=pretrained, **kwargs)


def vgg16_bn(model: SupervisedClassifierModel,
             x: tf.Tensor,
             pretrained: bool,
             **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        model: Model object
        x: Input Tensor
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg(model, x, cfg="D", batch_norm=True, pretrained=pretrained, **kwargs)

# def vgg11(pretrained=False, progress=True, **kwargs):
#     r"""VGG 11-layer model (configuration "A") from
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)
#
#
# def vgg11_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 11-layer model (configuration "A") with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)
#
#
# def vgg13(pretrained=False, progress=True, **kwargs):
#     r"""VGG 13-layer model (configuration "B")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)
#
#
# def vgg13_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 13-layer model (configuration "B") with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)
#
#
#
#
# def vgg19(pretrained=False, progress=True, **kwargs):
#     r"""VGG 19-layer model (configuration "E")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)
#
#
# def vgg19_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 19-layer model (configuration 'E') with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
