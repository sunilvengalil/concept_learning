from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST

cluster_column_name = "cluster_level_1"
cluster_column_name_2 = "cluster_level_2"
cluster_column_name_3 = "cluster_level_3"
num_clusters = 10
num_level_2_clusters = 10
num_units = [16, 16, 16, 16]
z_dim = 16
run_id = 14
experiment_name = "experiment_"
for i in range(len(num_units)):
    experiment_name = experiment_name + "_" + str(num_units[i])
create_split = False
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_ELBOW
env = "colab"

fully_convolutional = True
num_concepts = 20
strides = [2, 2, 2, 1, 1]
num_dense_layers = 0

# experiment_name = f"experiment_{num_units_str}"
_exp_config = ExperimentConfig(root_path=None,
                               num_decoder_layer=-1,
                               strides=strides,
                               z_dim=z_dim,
                               num_units=num_units,
                               num_cluster_config=num_cluster_config,
                               confidence_decay_factor=run_id,
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
                               activation_output_layer="SIGMOID",
                               learning_rate=1e-3,
                               env="colab",
                               num_dense_layers=num_dense_layers,
                               fully_convolutional=fully_convolutional
                               )
_exp_config.set_root_path(env)
_exp_config.check_and_create_directories(run_id)
model_type = MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST
dao = get_dao(_exp_config.dataset_name, _exp_config.split_name, _exp_config.num_val_samples)
