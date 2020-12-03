import tensorflow as tf
import argparse
from clearn.experiments.experiment import Experiment
import json
import os

from clearn.models.classify.classifier import ClassifierModel
from clearn.utils.data_loader import TrainValDataIterator
from clearn.utils.utils import show_all_variables
from clearn.config import ExperimentConfig
from clearn.config import RUN_ID

create_split = False
z_dim = 10
experiment_name = "semi_supervised_classification"

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
    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
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
    num_epochs = 10

    _config = ExperimentConfig(root_path="/Users/sunilv/concept_learning_exp",
                               num_decoder_layer=4,
                               z_dim=z_dim,
                               num_units=[64, 128, 32],
                               num_cluster_config=ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW,
                               confidence_decay_factor=5,
                               beta=5,
                               supervise_weight=150,
                               dataset_name="mnist",
                               split_name="Split_1",
                               model_name="VAE",
                               batch_size=64,
                               eval_interval=300,
                               name=experiment_name,
                               num_val_samples=128,
                               total_training_samples=60000,
                               manual_labels_config=TrainValDataIterator.USE_CLUSTER_CENTER,
                               reconstruction_weight=1,
                               activation_hidden_layer="RELU",
                               activation_output_layer="SIGMOID"
                               )
    _config.check_and_create_directories(RUN_ID)
    BATCH_SIZE = _config.BATCH_SIZE
    DATASET_NAME = _config.dataset_name
    _config.check_and_create_directories(RUN_ID, create=False)

    # TODO make this a configuration
    # to change output type from sigmoid to leaky relu, do the following
    # 1. In vae.py change the output layer type in decode()
    # 2. Change the loss function in build_model

    exp = Experiment(1, "VAE_MNIST", 128, _config, RUN_ID)

    print(exp.as_json())
    with open(_config.BASE_PATH + "config.json", "w") as config_file:
        json.dump(_config.as_json(), config_file)
    if create_split:
        train_val_data_iterator = TrainValDataIterator(exp.config.DATASET_ROOT_PATH,
                                                       shuffle=True,
                                                       stratified=True,
                                                       validation_samples=exp.num_validation_samples,
                                                       split_names=["train", "validation"],
                                                       split_location=exp.config.DATASET_PATH,
                                                       batch_size=exp.config.BATCH_SIZE)
    else:
        manual_annotation_file = os.path.join(_config.ANALYSIS_PATH,
                                              f"manual_annotation_epoch_{num_epochs - 1:.1f}.csv"
                                              )

        train_val_data_iterator = TrainValDataIterator.from_existing_split(exp.config.split_name,
                                                                           exp.config.DATASET_PATH,
                                                                           exp.config.BATCH_SIZE,
                                                                           manual_labels_config=exp.config.manual_labels_config,
                                                                           manual_annotation_file=manual_annotation_file)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = ClassifierModel(exp_config=_config,
                                sess=sess,
                                epoch=num_epochs,
                                batch_size=_config.BATCH_SIZE,
                                z_dim=_config.Z_DIM,
                                dataset_name=DATASET_NAME,
                                beta=_config.beta,
                                num_units_in_layer=_config.num_units,
                                train_val_data_iterator=train_val_data_iterator,
                                log_dir=exp.config.LOG_PATH,
                                checkpoint_dir=exp.config.TRAINED_MODELS_PATH,
                                result_dir=exp.config.PREDICTION_RESULTS_PATH,
                                supervise_weight=exp.config.supervise_weight,
                                reconstruction_weight=exp.config.reconstruction_weight,
                                reconstructed_image_dir=exp.config.reconstructed_images_path
                                )
        exp.model = model
        # show network architecture
        show_all_variables()

        exp.train(train_val_data_iterator)

        train_val_data_iterator.reset_counter("train")
        train_val_data_iterator.reset_counter("val")
        exp.encode_latent_vector(train_val_data_iterator, num_epochs, "train")

        train_val_data_iterator.reset_counter("train")
        train_val_data_iterator.reset_counter("val")
        exp.encode_latent_vector(train_val_data_iterator, num_epochs, "val")

