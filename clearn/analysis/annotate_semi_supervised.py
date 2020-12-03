from clearn.analysis.annotate import annotate
from clearn.config import ExperimentConfig
from clearn.utils.data_loader import TrainValDataIterator
import argparse

experiment_name = "semi_supervised_classification"
z_dim_range = [4, 5, 1]
num_epochs = 10
num_runs = 5
create_split = False
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
completed_runs = 4
#for z_dim in [21]:
for z_dim in [10]:
    for run_id in [100]:
        exp_config = ExperimentConfig(root_path="/Users/sunilv/concept_learning_exp",
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
        print(args.epoch, args.batch)
        annotate(exp_config, run_id, args.epoch, args.batch)
    completed_runs = 0