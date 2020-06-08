import argparse
import tensorflow as tf
from generative_models.vae import VAE
from utils.utils import check_folder
from utils.utils import show_all_variables

num_val_samples = 128
from common.data_loader import TrainValDataIterator
from config.exp_paths import DATASET_PATH
"""parsing and configuration"""

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='VAE',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN',
                                 'LSGAN', 'VAE', 'CVAE'],
                        help='The type of GAN', required=False)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA', 'documents'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=20, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        model = VAE(sess,
                    epoch=args.epoch,
                    batch_size=args.batch_size,
                    z_dim=args.z_dim,
                    dataset_name=args.dataset)
        # build graph
        model.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        train_val_data_iterator = TrainValDataIterator(DATASET_PATH,
                                                       shuffle=True,
                                                       stratified=True,
                                                       validation_samples=num_val_samples)
        model.train(train_val_data_iterator)
        print(" [*] Training finished!")

        # visualize learned generator
        #gan.visualize_results()

        print(" [*] Testing finished!")

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
    main()
