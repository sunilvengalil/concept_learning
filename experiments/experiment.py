import tensorflow as tf
import argparse
from analysis.encode_images import encode_images
import json

from generative_models.vae import VAE
from common.data_loader import TrainValDataIterator
from utils.utils import show_all_variables
from config import ExperimentConfig
create_split = False


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


"""checking arguments"""
def check_args(args):
    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args


class Experiment:
    def __init__(self, exp_id, name, num_val_samples, config1, run_id = None):
        if run_id is None:
            self.run_id = id
        else:
            self.run_id = run_id
        self.id = exp_id
        self.name = name
        self.num_validation_samples = num_val_samples
        self.config = config1

    def initialize(self, model=None):
        self.model =  model
        self.config.create_directories(self.run_id)

    def asJson(self):
        config_json = self.config.asJson()
        config_json["RUN_ID"] = self.run_id
        config_json["ID"] = self.id
        config_json["name"] = self.name
        config_json["NUM_VALIDATION_SAMPLES"] = self.num_validation_samples
        return config_json

    def train(self, train_val_data_iterator=None, create_split=False):
        if train_val_data_iterator is None:

            if create_split:
                train_val_data_iterator = TrainValDataIterator(self.config.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS,
                                                               shuffle=True,
                                                               stratified=True,
                                                               validation_samples=self.num_validtion_samples,
                                                               split_names=["train","validation"],
                                                               split_location=self.config.SPLIT_PATH,
                                                               batch_size=self.config.BATCH_SIZE)
            else:
                train_val_data_iterator = TrainValDataIterator.from_existing_split(self.config.split_name,
                                                                                   self.config.SPLIT_PATH,
                                                                                   self.config.BATCH_SIZE)
        self.model.train(train_val_data_iterator)
        print(" [*] Training finished!")

    def encode_latent_vector(self, train_val_data_iterator, dataset_type):
        encoded_df = encode_images(self.model,
                                   train_val_data_iterator,
                                   self.config,
                                   dataset_type)


def generate_image(z):
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        model = VAE

        # build graph
        model.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        model.restore_from_checkpoint()
        #print(" [*] Training finished!")

        # visualize learned generator
        batchsize=64
        #if z.shape[0] > 64:
        num_batches = z.shape[0]//64
        generated_images = []

        for i in range(num_batches):
            generated_images.append(model.generate_image(z[i*batchsize:i*batchsize+batchsize]))

        return generated_images

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    N_3 = 16
    N_2 = 128
    Z_DIM = 20
    run_id = 1

    ROOT_PATH = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
    _config = ExperimentConfig(ROOT_PATH, 4, Z_DIM, [64, N_2, N_3])

    BATCH_SIZE = _config.BATCH_SIZE
    DATASET_NAME = _config.dataset_name

    _config.check_and_create_directories(run_id, create=False)

    # TODO make this a configuration
    # to change output type from sigmod to leaky relu, do the following
    #1. In vae.py change the output layer type in decode()
    #2. Change the loss function in build_model

    exp = Experiment(1, "VAE_MNIST", 128, _config, run_id)

    _config.check_and_create_directories(run_id)
    #TODO if file exists verify the configuration are same. Othervise create new file with new timestamp
    print(exp.asJson())
    with open( _config.BASE_PATH + "config.json","w") as config_file:
        json.dump(_config.asJson(),config_file)
    if create_split:
        train_val_data_iterator = TrainValDataIterator(exp.config.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS,
                                                       shuffle=True,
                                                       stratified=True,
                                                       validation_samples=exp.num_validtion_samples,
                                                       split_names=["train", "validation"],
                                                       split_location=exp.config.SPLIT_PATH,
                                                       batch_size=exp.config.BATCH_SIZE)
    else:
        train_val_data_iterator = TrainValDataIterator.from_existing_split(exp.config.split_name,
                                                                           exp.config.SPLIT_PATH,
                                                                           exp.config.BATCH_SIZE)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = VAE(sess,
                    epoch=args.epoch,
                    batch_size=_config.BATCH_SIZE,
                    z_dim=_config.Z_DIM,
                    dataset_name=args.dataset,
                    beta=_config.beta,
                    num_units_in_layer=_config.num_units,
                    train_val_data_iterator=train_val_data_iterator,
                    log_dir=exp.config.LOG_PATH,
                    checkpoint_dir=exp.config.TRAINED_MODELS_PATH,
                    result_dir=exp.config.PREDICTION_RESULTS_PATH,
                    supervise_weight=exp.config.supervise_weight
                    )
        exp.model = model
        # show network architecture
        show_all_variables()

        exp.train(train_val_data_iterator)

        # num_batches_train = train_val_data_iterator.get_num_samples_train() // exp.config.BATCH_SIZE
        #
        # checkpoint_counter = model.load_from_checkpoint()
        # epochs_completed = int(checkpoint_counter / num_batches_train)
        # print("Number of epochs trained in current chekpoint", epochs_completed)

        train_val_data_iterator.reset_train_couner()
        train_val_data_iterator.reset_val_couner()
        exp.encode_latent_vector(train_val_data_iterator, "train")

        train_val_data_iterator.reset_train_couner()
        train_val_data_iterator.reset_val_couner()
        exp.encode_latent_vector(train_val_data_iterator, "val")