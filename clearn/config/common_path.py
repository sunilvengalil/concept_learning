from clearn.config import ExperimentConfig
BATCH_SIZE = 64
MODEL_NAME ="VAE"
SPLIT_NAME="Split_1"
DATASET_NAME = 'mnist'


def get_encoded_csv_file(exp_config:ExperimentConfig, epoch:int, train_val="train"):
    n_3 = exp_config.num_units[exp_config.num_decoder_layer - 2]
    n_2 = exp_config.num_units[exp_config.num_decoder_layer - 3]

    return f"z_{train_val}_{n_2}_{n_3}_{exp_config.Z_DIM}_epoch_{epoch}.csv"
