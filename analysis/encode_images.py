import tensorflow as tf
import os
import pandas as pd
from generative_models.vae import VAE
from config.analysis_paths import BATCH_SIZE, DATASET_NAME, \
    SPLIT_NAME, DATASET_PATH, Z_DIM, N_3, N_2,DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, \
    MODEL_NAME_WITH_CONFIG
from utils.utils import show_all_variables
from common.data_loader import TrainValDataIterator
from utils.utils import check_folder, get_latent_vector_column
from config.common_path import get_encoded_csv_file

epoch = 20

num_units_in_layer = [64, N_2, N_3, Z_DIM * 2]

MODEL_PATH = os.path.join(DATASET_PATH, MODEL_NAME_WITH_CONFIG)
SPLIT_PATH = os.path.join(DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, SPLIT_NAME + "/")
TRAINED_MODELS_PATH = os.path.join(MODEL_PATH, "trained_models/")
PREDICTION_RESULTS_PATH = os.path.join(MODEL_PATH, "prediction_results/")
LOG_PATH = os.path.join(MODEL_PATH, "logs/")
ANALYSIS_PATH = os.path.join(MODEL_PATH, "analysis/")
check_folder(ANALYSIS_PATH)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #   declare instance for VAE
    model = VAE(sess,
                epoch=epoch,
                batch_size=BATCH_SIZE,
                z_dim=Z_DIM,
                dataset_name=DATASET_NAME,
                log_dir=LOG_PATH,
                checkpoint_dir=TRAINED_MODELS_PATH,
                result_dir=PREDICTION_RESULTS_PATH,
                num_units_in_layer=num_units_in_layer
                )

#   build graph
    model.build_model()
#   show network architecture
    show_all_variables()

#   read the training images
    train_val_data_iterator = TrainValDataIterator.from_existing_split(SPLIT_NAME,
                                                                       SPLIT_PATH,
                                                                       BATCH_SIZE)

    num_batches_train = train_val_data_iterator.get_num_samples_train() // BATCH_SIZE

    checkpoint_counter = model.load_from_checkpoint()
    epochs_completed = int(checkpoint_counter / num_batches_train)
    print("Number of epochs trained in current chekpoint", epochs_completed)

    # cluster latent vector learned generator
    encoded_df = None
    while train_val_data_iterator.has_next_train():
        batch_images = train_val_data_iterator.get_next_batch_train()
        if batch_images.shape[0] < BATCH_SIZE:
            train_val_data_iterator.reset_train_couner()
            break

        mu, sigma, z = model.encode(batch_images)
        mean_col_names, sigma_col_names, z_col_names = get_latent_vector_column(Z_DIM)

        temp_df1 = pd.DataFrame(mu, columns=mean_col_names)
        temp_df2 = pd.DataFrame(sigma, columns=sigma_col_names)
        temp_df3 = pd.DataFrame(z, columns=z_col_names)
        temp_df = pd.concat([temp_df1, temp_df2, temp_df3], axis=1)
        if encoded_df is not None:
            encoded_df = pd.concat([encoded_df, temp_df])
        else:
            encoded_df = temp_df
    print(ANALYSIS_PATH)
    output_csv_file = get_encoded_csv_file(N_2,N_3,Z_DIM,"train")
    encoded_df.to_csv(os.path.join(ANALYSIS_PATH, output_csv_file), index=False)
