import tensorflow as tf
import argparse
from clearn.experiments.experiment import initialize_model_train_and_get_features, VAAL_ARCHITECTURE_FOR_CIFAR
from clearn.config import ExperimentConfig


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
    num_epochs = 20
    create_split = True
    z_dim = 32
    experiment_name = "Experiment_3"
    num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
    run_id = 101
    train_val_data_iterator = None
    tf.reset_default_graph()

    for num_units in [[64, 128, 128, 128],
                      [64, 128, 128, 64],
                      [64, 128, 128, 32],
                      [64, 128, 64, 64],
                      [64, 128, 64, 32],
                      [64, 128, 32, 64],
                      [64, 128, 32, 32]][3:4]:
        train_val_data_iterator, exp_config, model = initialize_model_train_and_get_features(
            experiment_name=experiment_name,
            z_dim=z_dim,
            run_id=run_id,
            create_split=create_split,
            num_epochs=num_epochs,
            num_cluster_config=num_cluster_config,
            manual_labels_config=ExperimentConfig.USE_ACTUAL,
            supervise_weight=1,
            beta=0,
            reconstruction_weight=0,
            num_units=num_units,
            save_reconstructed_images=False,
            split_name="split_1",
            train_val_data_iterator=train_val_data_iterator,
            num_val_samples=5000,
            learning_rate=0.001,
            dataset_name="cifar_10",
            activation_output_layer="LINEAR",
            write_predictions=False,
            num_decoder_layer=5,
            model_type=VAAL_ARCHITECTURE_FOR_CIFAR
            )
        tf.reset_default_graph()
