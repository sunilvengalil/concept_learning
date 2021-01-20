from clearn.analysis.combine_keys import combine_keys
from clearn.config import ExperimentConfig
from clearn.utils.data_loader import TrainValDataIterator
import argparse

create_split = False
z_dim = 10
experiment_name = "un_supervised_classification"
ROOT_PATH = "/Users/sunilv/concept_learning_exp"
z_dim_range = [1, 5, 1]
num_epochs = 10
num_runs = 3
completed_z_dims = 0


def parse_args():
    desc = "Start annotation of images"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()


args = parse_args()
start_epoch = args.epoch
start_batch_id = args.batch
completed_runs = 1
for z_dim in [1]:
#for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
    if z_dim <= completed_z_dims:
        continue
    for run_id in [0]:
        exp_config = ExperimentConfig(root_path=ROOT_PATH,
                                      num_decoder_layer=4,
                                      z_dim=z_dim,
                                      num_units=[64, 128, 32],
                                      num_cluster_config=None,
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
                                      manual_labels_config=TrainValDataIterator.ExperimentConfig,
                                      reconstruction_weight=1,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer="SIGMOID"
                                      )

        print(args.epoch, args.batch)
        combine_keys(exp_config, run_id)
    completed_runs = 0