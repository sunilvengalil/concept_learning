import numpy as np


def decode(model, z, BATCH_SIZE):
    # TODO remove this hardcoding
    feature_dimension = [len(z), 28, 28, 1]
    reconstructed_images = np.zeros(feature_dimension)
    num_latent_vectors = z.shape[0]
    num_batches = num_latent_vectors // BATCH_SIZE
    for batch_num in range(num_batches):
        decoded_images = model.decode(z[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE])
        reconstructed_images[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE] = decoded_images
    left_out = num_latent_vectors % BATCH_SIZE
    if left_out != 0:
        last_batch = np.zeros([BATCH_SIZE, z.shape[1]])
        last_batch[0:left_out, :] = z[num_batches * BATCH_SIZE:]
        decoded_images = model.decode(last_batch)
        reconstructed_images[num_batches * BATCH_SIZE:] = decoded_images[0:left_out]

    return reconstructed_images


def encode(model, images, batch_size, z_dim):
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


def encode_and_get_features(model, images, batch_size):
    num_images = images.shape[0]
    num_batches = num_images // batch_size
    batch_num = 0
    mus = []
    sigmas =[]
    zs =[]
    dense2_ens = []
    reshapeds = []
    conv2_ens= []
    conv1_ens = []
    if num_batches >= 1:

        # Run for first batch to get the dimensions
        mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = model.encode_and_get_features(images[batch_num * batch_size: (batch_num + 1) * batch_size])
        mus.append(mu)
        sigmas.append(sigma)
        zs.append(z)
        dense2_ens.append(dense2_en)
        reshapeds.append(reshaped)
        conv2_ens.append(conv2_en)
        conv1_ens.append(conv1_en)

        for batch_num in range(1,num_batches):
            mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = model.encode_and_get_features(
                images[batch_num * batch_size: (batch_num + 1) * batch_size])
            mus.append(mu)
            sigmas.append(sigma)
            zs.append(z)
            dense2_ens.append(dense2_en)
            reshapeds.append(reshaped)
            conv2_ens.append(conv2_en)
            conv1_ens.append(conv1_en)

    left_out = num_images % batch_size
    if left_out > 0:
        # TODO remove this hardcoding
        feature_dimension = [batch_size, 28, 28, 1]
        last_batch = np.zeros(feature_dimension)
        last_batch[0:left_out] = images[num_batches * batch_size:]
        mu, sigma, z, dense2_en, reshaped, conv2_en, conv1_en = model.encode_and_get_features(
            last_batch)
        mus.append(mu)
        sigmas.append(sigma)
        zs.append(z)
        dense2_ens.append(dense2_en)
        reshapeds.append(reshaped)
        conv2_ens.append(conv2_en)
        conv1_ens.append(conv1_en)

    return mus, sigmas, zs, dense2_ens, reshapeds, conv2_ens, conv1_ens
