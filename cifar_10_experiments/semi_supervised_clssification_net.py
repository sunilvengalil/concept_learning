from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10

experiment_name = "Experiment_6"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim = 32
learning_rate = 0.001
num_epochs = 20 
num_runs = 1
create_split = True
completed_z_dims = 0
completed_runs = 0
run_id = 1
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
num_units = [64, 128, 64, 64]
# num_units = [128, 256, 512, 1024]
train_val_data_iterator = None
beta = 0
supervise_weight = 0
dataset_name = "cifar_10"
split_name = "split_1"
dao = get_dao(dataset_name, split_name, 128)
num_epochs_completed = 15


initialize_model_train_and_get_features(experiment_name=experiment_name,
                                        z_dim=z_dim,
                                        run_id=run_id,
                                        create_split=create_split,
                                        num_epochs=num_epochs,
                                        num_cluster_config=num_cluster_config,
                                        manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                        supervise_weight=0.05,
                                        beta=0,
                                        reconstruction_weight=1,
                                        model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
                                        num_units=num_units,
                                        save_reconstructed_images=True,
                                        split_name="split_1",
                                        train_val_data_iterator=train_val_data_iterator,
                                        num_val_samples=128,
                                        learning_rate=0.0001,
                                        dataset_name="cifar_10",
                                        activation_output_layer="LINEAR",
                                        num_decoder_layer=len(num_units) + 1,
                                        write_predictions=True,
                                        seed=547,
                                        num_epochs_completed=num_epochs_completed
                                        )
