import os
ROOT_PATH = "/Users/sunilkumar/concept_learning/image_classification/"
BATCH_SIZE = 64
MODEL_NAME ="VAE"
SPLIT_NAME="Split_1"
DATASET_NAME = 'mnist'

def get_base_path(z_dim, n_3, n_2):
    return os.path.join(ROOT_PATH, "Exp_{:02d}_{:03}_{:03d}/".format(z_dim, n_3, n_2))

def get_encoded_csv_file(N_2, N_3, Z_DIM,train_val="train"):
    return "z_{}_{}_{}_{}.csv".format(train_val,N_2, N_3, Z_DIM)
