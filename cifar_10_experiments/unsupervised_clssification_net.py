from clearn.config import ExperimentConfig
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10

experiment_name = "Experiment_4"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim = 32
learning_rate = 0.001
num_epochs = 20
num_runs = 1
create_split = True
completed_z_dims = 0
# for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
completed_runs = 0
run_id = 1
# for run_id in range(num_runs):
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
num_units = [32, 64, 64, 64]
train_val_data_iterator = None
initialize_model_train_and_get_features(experiment_name=experiment_name,
                                        z_dim=z_dim,
                                        run_id=run_id,
                                        create_split=create_split,
                                        num_epochs=num_epochs,
                                        num_cluster_config=num_cluster_config,
                                        manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                        supervise_weight=1,
                                        beta=5,
                                        reconstruction_weight=1,
                                        model_type=MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,
                                        num_units=num_units,
                                        save_reconstructed_images=False,
                                        split_name="split_1",
                                        train_val_data_iterator=train_val_data_iterator,
                                        num_val_samples=128,
                                        learning_rate=0.005,
                                        dataset_name="cifar_10",
                                        activation_output_layer="LINEAR",
                                        num_decoder_layer=5,
                                        write_predictions=False
                                        )
