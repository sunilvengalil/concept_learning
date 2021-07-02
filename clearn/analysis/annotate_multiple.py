from clearn.analysis.annotate import annotate
from clearn.config import ExperimentConfig
import argparse

create_split = False
experiment_name = "Experiment_1"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim_range = [5, 15, 2]
learning_rate = 0.001
num_epochs = 10
num_runs = 5
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
num_cluster_config = None
for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
    print(f"Starting annotation for z_dim {z_dim}")
    for run_id in range(num_runs):
        exp_config = ExperimentConfig(root_path=root_path,
                                      num_decoder_layer=4,
                                      z_dim=z_dim,
                                      num_units=[64, 128, 32],
                                      num_cluster_config=num_cluster_config,
                                      confidence_decay_factor=5,
                                      beta=5,
                                      supervise_weight=150,
                                      dataset_name="mnist",
                                      split_name="Split_1",
                                      model_name="VAE",
                                      batch_size=64,
                                      eval_interval_in_epochs=1,
                                      name=experiment_name,
                                      num_val_samples=128,
                                      total_training_samples=60000,
                                      manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                      reconstruction_weight=1,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer="SIGMOID"
                                      )
        print(args.epoch, args.batch)
        annotate(exp_config, run_id, args.epoch, args.batch)
