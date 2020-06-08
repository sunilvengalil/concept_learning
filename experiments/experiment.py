import tensorflow as tf
import os
import argparse

from generative_models.vae import VAE
from config.exp_paths import BASE_PATH, DATASET_PATH,\
    DATASET_ROOT_PATH, \
    DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, SPLIT_NAME, BATCH_SIZE, Z_DIM, N_3, N_2,\
    MODEL_NAME_WITH_CONFIG
from common.data_loader import TrainValDataIterator
from utils.utils import show_all_variables
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
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=Z_DIM, help='Dimension of noise vector')
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
    def __init__(self, exp_id, name, num_val_samples):
        self.id = exp_id
        self.name = name
        self.num_validtion_samples = num_val_samples

    def create_directories(self, model):
        if model is not None:
            self.model = model

            self.MODEL_PATH = os.path.join(DATASET_PATH, MODEL_NAME_WITH_CONFIG)
            self.SPLIT_PATH = os.path.join(DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, SPLIT_NAME + "/")
            # self.TRAINED_MODELS_PATH = os.path.join(self.MODEL_PATH, "trained_models/")
            # self.PREDICTION_RESULTS_PATH = os.path.join(self.MODEL_PATH, "prediction_results/")
            # self.LOG_PATH = os.path.join(self.MODEL_PATH,"logs/")
            self.TRAINED_MODELS_PATH = os.path.join(self.MODEL_PATH, "trained_models/")
            self.PREDICTION_RESULTS_PATH = os.path.join(self.MODEL_PATH, "prediction_results/")
            self.LOG_PATH = os.path.join(self.MODEL_PATH,"logs/")

            if not os.path.isdir(BASE_PATH):
                print("Creating directory{}".format(BASE_PATH))
                os.mkdir(BASE_PATH)
            if not os.path.isdir(DATASET_ROOT_PATH):
                print("Creating directory{}".format(DATASET_ROOT_PATH))
                os.mkdir(DATASET_ROOT_PATH)
            if not os.path.isdir(DATASET_PATH):
                print("Creating directory{}".format(DATASET_PATH))
                os.mkdir(DATASET_PATH)
            if not os.path.isdir(self.MODEL_PATH):
                print("Creating directory{}".format(self.MODEL_PATH))
                os.mkdir(self.MODEL_PATH)

            if not os.path.isdir(self.SPLIT_PATH):
                print("Creating directory{}".format(self.SPLIT_PATH))
                os.mkdir(self.SPLIT_PATH)

            if not os.path.isdir(self.TRAINED_MODELS_PATH):
                os.mkdir(self.TRAINED_MODELS_PATH)
            if not os.path.isdir(self.PREDICTION_RESULTS_PATH):
                os.mkdir(self.PREDICTION_RESULTS_PATH)
            if not os.path.isdir(self.LOG_PATH):
                print("Creating directory{}".format(BASE_PATH))
                os.mkdir(self.LOG_PATH)

    def initialze(self,model = None):
        self.create_directories(model)

    def train(self):
        # build graph
        model.build_model()

        # show network architecture
        show_all_variables()

        if create_split:
            train_val_data_iterator = TrainValDataIterator(DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS,
                                                           shuffle=True,
                                                           stratified=True,
                                                           validation_samples=self.num_validtion_samples,
                                                           split_names=["train","validation"],
                                                           split_location=self.SPLIT_PATH,
                                                           batch_size=BATCH_SIZE)
        else:
            train_val_data_iterator = TrainValDataIterator.from_existing_split(SPLIT_NAME,
                                                                               self.SPLIT_PATH,
                                                                               BATCH_SIZE)

        model.train(train_val_data_iterator)
        print(" [*] Training finished!")

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
    if args is None:
      exit()
    exp = Experiment(1, "VAE_MNIST",128)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = VAE(sess,
                    epoch=args.epoch,
                    batch_size=args.batch_size,
                    z_dim=args.z_dim,
                    dataset_name=args.dataset,
                    num_units_in_layer=[64, N_2, N_3, args.z_dim * 2])
        exp.initialze(model)
        model.set_result_directories(log_dir=exp.LOG_PATH,
                    checkpoint_dir=exp.TRAINED_MODELS_PATH,
                    result_dir=exp.PREDICTION_RESULTS_PATH)

        exp.train()