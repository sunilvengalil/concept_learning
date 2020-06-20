import tensorflow as tf
import os
from generative_models.vae import VAE
from config.analysis_paths import BATCH_SIZE, DATASET_NAME, \
    SPLIT_NAME, DATASET_PATH, Z_DIM,N_2, N_3,\
    DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS,MODEL_NAME_WITH_CONFIG
from utils.utils import show_all_variables
from common.data_loader import TrainValDataIterator
from utils.utils import segregate_images_by_label,check_folder,get_latent_vector_column
import numpy as np
from config.common_path import get_encoded_csv_file

epoch = 5
import pandas as pd

num_units_in_layer = [64, N_2, N_3, Z_DIM * 2]

MODEL_PATH = os.path.join(DATASET_PATH, MODEL_NAME_WITH_CONFIG)
SPLIT_PATH = os.path.join(DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, SPLIT_NAME + "/")
TRAINED_MODELS_PATH = os.path.join(MODEL_PATH, "trained_models/")
PREDICTION_RESULTS_PATH = os.path.join(MODEL_PATH, "prediction_results/")
LOG_PATH = os.path.join(MODEL_PATH, "logs/")
ANALYSIS_PATH = os.path.join(MODEL_PATH, "analysis/")
check_folder(ANALYSIS_PATH)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # model = VAE(sess,
    #             epoch=epoch,
    #             batch_size=BATCH_SIZE,
    #             z_dim=Z_DIM,
    #             dataset_name=DATASET_NAME,
    #             log_dir=LOG_PATH,
    #             checkpoint_dir=TRAINED_MODELS_PATH,
    #             result_dir=PREDICTION_RESULTS_PATH)
    #
    # # build graph
    # model.build_model()
    # # show network architecture
    # show_all_variables()


    train_val_data_iterator = TrainValDataIterator.from_existing_split(SPLIT_NAME,
                                                                       SPLIT_PATH,BATCH_SIZE)
    num_batches_train = train_val_data_iterator.get_num_samples_train() // BATCH_SIZE

    #checkpoint_counter = model.load_from_checkpoint()
    #epochs_completed = (int)(checkpoint_counter / num_batches_train)
    #print("Number of epochs completed", epochs_completed)

    # visualize learned generator
    #TODO change this to get a random batch.
    #TODO Also change to do analysis on training images also
    images_by_label = segregate_images_by_label(train_val_data_iterator)

    df = None
    for label, images in images_by_label.items():
        if len(images) == 0:
            print("No images for label {} in data ".format(label))
            continue

        _labels = np.zeros(BATCH_SIZE)
        _labels[:len(images)] = label
        number_of_bank_images_to_be_added = BATCH_SIZE - len(images)
        _labels[len(images):] = -1
        for i in range(number_of_bank_images_to_be_added):
            images.append(np.zeros([28, 28, 1]))

        mu, sigma, z = model.encode(images)
        reconstructed_images = model.generate_image(mu, sigma)

        mean_col_names, sigma_col_names, z_col_names = get_latent_vector_column(Z_DIM)

        temp_df1 = pd.DataFrame(mu, columns=mean_col_names)
        temp_df2 = pd.DataFrame(sigma, columns=sigma_col_names)
        temp_df3 = pd.DataFrame(z, columns=z_col_names)
        temp_df = pd.concat([temp_df1, temp_df2, temp_df3], axis=1)

        temp_df["label"] = _labels
        if df is not None:
            df = pd.concat([df, temp_df])
        else:
            df =temp_df

        print(mu.shape, sigma.shape, z.shape, reconstructed_images.shape)
    filename = get_encoded_csv_file(N_2,N_3,Z_DIM,"val")
    print(filename)
    df.to_csv(os.path.join(ANALYSIS_PATH, filename),index=False)

