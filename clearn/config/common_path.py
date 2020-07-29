BATCH_SIZE = 64
MODEL_NAME ="VAE"
SPLIT_NAME="Split_1"
DATASET_NAME = 'mnist'


def get_encoded_csv_file(n_2, n_3, z_dim, epoch, train_val="train"):
    return f"z_{train_val}_{n_2}_{n_3}_{z_dim}_epoch_{epoch}.csv"
