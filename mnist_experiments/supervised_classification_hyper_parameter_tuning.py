import tensorflow as tf
import argparse
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_SUPERVISED_CLASSIFIER
from clearn.config import ExperimentConfig

create_split = False
z_dim = 10
experiment_name = "Experiment_2"


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='VAE',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN',
                                 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN',
                                 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=False)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'celebA', 'documents'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=10, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


def check_args(_args):
    # --epoch
    assert _args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert _args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert _args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return _args


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    num_epochs = 40
    num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
    run_id = 1
    z_dim_range = [5, 30, 2]
    train_val_data_iterator = None
    for num_units in [[64, 128, 32],
                      [32, 64, 16],
                      [16, 32, 8],
                      [8, 16, 4],
                      [4, 8, 2],
                      [2, 4, 1], [1, 2, 1]][6:]:
        for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):

            train_val_data_iterator, _, _ = initialize_model_train_and_get_features(experiment_name=experiment_name,
                                                                                    z_dim=z_dim,
                                                                                    run_id=run_id,
                                                                                    create_split=create_split,
                                                                                    num_epochs=num_epochs,
                                                                                    num_cluster_config=num_cluster_config,
                                                                                    manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                                                                    supervise_weight=1,
                                                                                    beta=0,
                                                                                    reconstruction_weight=0,
                                                                                    model_type=MODEL_TYPE_SUPERVISED_CLASSIFIER,
                                                                                    num_decoder_layer=3,
                                                                                    num_units=num_units[0:2],
                                                                                    save_reconstructed_images=False,
                                                                                    split_name="Split_70_30",
                                                                                    train_val_data_iterator=train_val_data_iterator,
                                                                                    num_val_samples=-1,
                                                                                    write_predictions=False
                                                                                    )
            tf.reset_default_graph()
