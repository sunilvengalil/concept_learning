import numpy as np

from clearn.models.generative_model import GenerativeModel


def decode(model: GenerativeModel, z, batch_size):
    """
    z =
    """
    feature_shape = model.dao.image_shape
    feature_dimension = [len(z), feature_shape[0], feature_shape[1], feature_shape[2]]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // batch_size
    for batch_num in range(num_batches):
        decoded_images = model.decode(z[batch_num * batch_size: (batch_num + 1) * batch_size])
        reconstructed_images[batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images
    left_out = num_latent_vectors % batch_size
    if left_out != 0:
        last_batch = np.zeros([batch_size, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * batch_size:]
        decoded_images = model.decode(last_batch)
        reconstructed_images[num_batches * batch_size:] = decoded_images[0:left_out]

    return reconstructed_images


def decode_l3(model: GenerativeModel, z: np.ndarray, batch_size: int):
    """
    z =
    """
    feature_dimension = [len(z), model.dao.image_shape[0], model.dao.image_shape[1], model.dao.image_shape[2]]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // batch_size
    for batch_num in range(num_batches):
        decoded_images = model.decode_l3(z[batch_num * batch_size: (batch_num + 1) * batch_size])
        reconstructed_images[batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images
    left_out = num_latent_vectors % batch_size
    if left_out != 0:
        last_batch = np.zeros([batch_size, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * batch_size:]
        decoded_images = model.decode_l3(last_batch)
        reconstructed_images[num_batches * batch_size:] = decoded_images[0:left_out]

    return reconstructed_images


def decode_layer1(model, z, batch_size):
    feature_dimension = [len(z), model.exp_config.num_units[2]]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // batch_size
    for batch_num in range(num_batches):
        decoded_images = model.decode_layer1(z[batch_num * batch_size: (batch_num + 1) * batch_size])
        reconstructed_images[batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images
    left_out = num_latent_vectors % batch_size
    if left_out != 0:
        last_batch = np.zeros([batch_size, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * batch_size:]
        decoded_images = model.decode_layer1(last_batch)
        reconstructed_images[num_batches * batch_size:] = decoded_images[0:left_out]

    return reconstructed_images


def classify_images(model, images, batch_size, num_classes):
    num_images = images.shape[0]
    num_batches = num_images // batch_size
    logits = np.zeros([num_batches * batch_size, num_classes])
    for batch_num in range(num_batches):
        _logits = model.classify(images[batch_num * batch_size: (batch_num + 1) * batch_size])[0]
        logits[batch_num * batch_size: (batch_num + 1) * batch_size] = _logits
    left_out = num_images % batch_size
    if left_out > 0:
        feature_dimension = [batch_size, model.dao.image_shape[0], model.dao.image_shape[1], model.dao.image_shape[2]]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        _logits = model.classify(last_batch)[0]
        logits[num_batches * batch_size:] = _logits[0:left_out]

    return logits


def encode(model, images, batch_size, z_dim):
    """
    Encode the given set of images using the model provided
    :param model: Model Object
    :param images: `ndarray` of images
    :param batch_size: batch_size to be used while encoding
    :param z_dim: latent vector dimension
    :return: `ndarray` of latent vectors corresponding  to images
    """
    latent_vectors = np.zeros([len(images), z_dim])
    num_images = images.shape[0]
    num_batches = num_images // batch_size
    for batch_num in range(num_batches):
        mu, sigma, z, _ = model.encode(images[batch_num * batch_size: (batch_num + 1) * batch_size])
        latent_vectors[batch_num * batch_size: (batch_num + 1) * batch_size] = z
    left_out = num_images % batch_size
    if left_out != 0:
        feature_dimension = [batch_size, model.dao.image_shape[0], model.dao.image_shape[1], model.dao.image_shape[2]]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        mu, sigma, z, _ = model.encode(last_batch)
        latent_vectors[num_batches * batch_size:] = z[0:left_out]
    return latent_vectors


def decode_and_get_features(model: GenerativeModel, z: np.ndarray, batch_size: int, layer_num=None, feature_num=None):
    feature_dimension = [len(z), model.dao.image_shape[0], model.dao.image_shape[1], model.dao.image_shape[2]]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // batch_size
    features_dict = dict()
    for batch_num in range(num_batches):
        feature_names, decoded_images_and_features = model.decode_and_get_features(z[batch_num * batch_size: (batch_num + 1) * batch_size],
                                                                                   layer_num, feature_num)
        reconstructed_images[batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images_and_features[0]
        for i, feature_name in enumerate(feature_names):
            if feature_name not in features_dict:
                features_dict[feature_name] = np.zeros([len(z),
                                                       decoded_images_and_features[i + 1].shape[1],
                                                       decoded_images_and_features[i + 1].shape[2],
                                                       decoded_images_and_features[i + 1].shape[3]]
                                                       )
            features_dict[feature_name][batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images_and_features[i + 1]

    left_out = num_latent_vectors % batch_size
    if left_out > 0:
        last_batch = np.zeros([batch_size, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * batch_size:]
        feature_names, decoded_images_and_features = model.decode_and_get_features(last_batch, layer_num, feature_num)
        reconstructed_images[num_batches * batch_size:] = decoded_images_and_features[0][0:left_out]
        for i, feature_name in enumerate(feature_names):
            if feature_name not in features_dict:
                features_dict[feature_name] = np.zeros([len(z),
                                                       decoded_images_and_features[i + 1].shape[1],
                                                       decoded_images_and_features[i + 1].shape[2],
                                                       decoded_images_and_features[i + 1].shape[3]]
                                                       )
            features_dict[feature_name][num_batches * batch_size:] = decoded_images_and_features[i + 1][0:left_out]

    return feature_names, reconstructed_images, features_dict


def encode_and_get_features(model: GenerativeModel,
                            images, batch_size, z_dim):
    num_images = images.shape[0]
    num_batches = num_images // batch_size
    mus = np.zeros([len(images), z_dim])
    sigmas = np.zeros([len(images), z_dim])
    latent_vectors = np.zeros([len(images), z_dim])
    features_dict = dict()

    for batch_num in range(num_batches):
        feature_names, mu, sigma, z, encoded_features = model.encode_and_get_features(
            images[batch_num * batch_size: (batch_num + 1) * batch_size])
        mus[batch_num * batch_size: (batch_num + 1) * batch_size] = mu
        sigmas[batch_num * batch_size: (batch_num + 1) * batch_size] = sigma
        latent_vectors[batch_num * batch_size: (batch_num + 1) * batch_size] = z
        for i, feature_name in enumerate(feature_names):
            if feature_name not in features_dict:
                features_dict[feature_name] = np.zeros([len(z),
                                                       encoded_features[i].shape[1],
                                                       encoded_features[i].shape[2],
                                                       encoded_features[i].shape[3]]
                                                       )
            features_dict[feature_name][batch_num * batch_size: (batch_num + 1) * batch_size] = encoded_features[i]

    left_out = num_images % batch_size
    if left_out > 0:
        feature_dimension = [batch_size, model.dao.image_shape[0], model.dao.image_shape[1], model.dao.image_shape[2]]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        feature_names, mu, sigma, z, encoded_features = model.encode_and_get_features(last_batch)
        mus[num_batches * batch_size:] = mu[0:left_out]
        sigmas[num_batches * batch_size:] = sigma[0:left_out]
        latent_vectors[num_batches * batch_size:] = z[0:left_out]
        for i, feature_name in enumerate(feature_names):
            if feature_name not in features_dict:
                features_dict[feature_name] = np.zeros([len(z),
                                                       encoded_features[i].shape[1],
                                                       encoded_features[i].shape[2],
                                                       encoded_features[i].shape[3]]
                                                       )
            features_dict[feature_name][num_batches * batch_size:] = encoded_features[i]

    return mus, sigmas, latent_vectors, features_dict
