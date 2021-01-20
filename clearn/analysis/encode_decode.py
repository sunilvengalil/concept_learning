import numpy as np


def decode(model, z, batch_size):
    """
    z =
    """
    # TODO remove this hard-coding
    feature_dimension = [len(z), 28, 28, 1]
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


def decode_l3(model, z, batch_size):
    """
    z =
    """
    # TODO remove this hard-coding
    feature_dimension = [len(z), 28, 28, 1]
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
    # TODO remove this hard-coding
    feature_dimension = [len(z), 32]
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
        # TODO remove this hard-coding
        feature_dimension = [batch_size, 28, 28, 1]
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
        mu, sigma, z = model.encode(images[batch_num * batch_size: (batch_num + 1) * batch_size])
        latent_vectors[batch_num * batch_size: (batch_num + 1) * batch_size] = z
    left_out = num_images % batch_size
    if left_out != 0:
        # TODO remove this hard coding
        feature_dimension = [batch_size, 28, 28, 1]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        mu, sigma, z = model.encode(last_batch)
        latent_vectors[num_batches * batch_size:] = z[0:left_out]
    return latent_vectors


def decode_and_get_features(model, z, batch_size):
    # TODO remove this hard-coding
    feature_dimension = [len(z), 28, 28, 1]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // batch_size
    dense1_des = []
    dense2_des = []
    reshaped_des = []
    deconv1_des = []
    for batch_num in range(num_batches):
        decoded_images, dense1_de, dense2_de, reshaped_de, deconv1_de = model.decode_and_get_features(z[batch_num * batch_size: (batch_num + 1) * batch_size])
        reconstructed_images[batch_num * batch_size: (batch_num + 1) * batch_size] = decoded_images
        dense1_des.append(dense1_de)
        dense2_des.append(dense2_de)
        reshaped_des.append(reshaped_de)
        deconv1_des.append(deconv1_de)

    left_out = num_latent_vectors % batch_size
    if left_out > 0:
        last_batch = np.zeros([batch_size, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * batch_size:]
        decoded_images, dense1_de, dense2_de, reshaped_de, deconv1_de = model.decode_and_get_features(last_batch)
        reconstructed_images[num_batches * batch_size:] = decoded_images[0:left_out]
        dense1_des.append(dense1_de)
        dense2_des.append(dense2_de)
        reshaped_des.append(reshaped_de)
        deconv1_des.append(deconv1_de)
    return reconstructed_images, dense1_des, dense2_des, reshaped_des, deconv1_des


def encode_and_get_features(model, images, batch_size, z_dim):
    num_images = images.shape[0]
    num_batches = num_images // batch_size
    mus = np.zeros([len(images), z_dim])
    sigmas = np.zeros([len(images), z_dim])
    latent_vectors = np.zeros([len(images), z_dim])
    # TODO remove the hard coding of feature dimension. Instead, ddd an api to get the dimension
    dense2_ens = np.zeros([len(images), 32])
    reshapeds = []
    conv2_ens = []
    conv1_ens = []

    for batch_num in range(num_batches):
        mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = model.encode_and_get_features(
            images[batch_num * batch_size: (batch_num + 1) * batch_size])
        mus[batch_num * batch_size: (batch_num + 1) * batch_size] = mu
        sigmas[batch_num * batch_size: (batch_num + 1) * batch_size] = sigma
        latent_vectors[batch_num * batch_size: (batch_num + 1) * batch_size] = z
        dense2_ens[batch_num * batch_size: (batch_num + 1) * batch_size] = dense2_en
        reshapeds.append(reshaped)
        conv2_ens.append(conv2_en)
        conv1_ens.append(conv1_en)

    left_out = num_images % batch_size
    if left_out > 0:
        # TODO remove this hard-coding
        feature_dimension = [batch_size, 28, 28, 1]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = model.encode_and_get_features(last_batch)
        mus[num_batches * batch_size:] = mu[0:left_out]
        sigmas[num_batches * batch_size:] = sigma[0:left_out]
        latent_vectors[num_batches * batch_size:] = z[0:left_out]
        dense2_ens[num_batches * batch_size:] = dense2_en[0:left_out]
        reshapeds.append(reshaped)
        conv2_ens.append(conv2_en)
        conv1_ens.append(conv1_en)

    return mus, sigmas, latent_vectors, dense2_ens, reshapeds, conv2_ens, conv1_ens
