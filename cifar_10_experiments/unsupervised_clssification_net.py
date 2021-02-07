from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10

experiment_name = "Experiment_4"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim = 32
learning_rate = 0.001
num_epochs = 100
num_runs = 1
create_split = True
completed_z_dims = 0
completed_runs = 0
run_id = 3
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
num_units = [64, 128, 64, 32]
# num_units = [128, 256, 512, 1024]
train_val_data_iterator = None
beta = 0
supervise_weight = 0
dataset_name = "cifar_10"
split_name = "split_1"
dao = get_dao(dataset_name, split_name)

# exp_config = ExperimentConfig(root_path=root_path,
#                               num_decoder_layer=len(num_units) + 1,
#                               z_dim=z_dim,
#                               num_units=num_units,
#                               num_cluster_config=num_cluster_config,
#                               confidence_decay_factor=5,
#                               beta=beta,
#                               supervise_weight=supervise_weight,
#                               dataset_name=dataset_name,
#                               split_name=split_name,
#                               model_name="VAE",
#                               batch_size=64,
#                               eval_interval=300,
#                               name=experiment_name,
#                               num_val_samples=128,
#                               total_training_samples=dao.number_of_training_samples,
#                               manual_labels_config=ExperimentConfig.USE_ACTUAL,
#                               reconstruction_weight=1,
#                               activation_hidden_layer="RELU",
#                               activation_output_layer="LINEAR",
#                               save_reconstructed_images=True,
#                               learning_rate=learning_rate,
#                               run_evaluation_during_training=True,
#                               write_predictions=False,
#                               seed=547
#                               )
# exp_config.check_and_create_directories(run_id, create=True)
#
# train_val_data_iterator = TrainValDataIterator(exp_config.DATASET_ROOT_PATH,
#                                                shuffle=True,
#                                                stratified=True,
#                                                validation_samples=exp_config.num_val_samples,
#                                                split_names=["train", "validation"],
#                                                split_location=exp_config.DATASET_PATH,
#                                                batch_size=exp_config.BATCH_SIZE,
#                                                manual_labels_config=exp_config.manual_labels_config,
#                                                manual_annotation_file=None,
#                                                dao=dao,
#                                                seed=exp_config.seed)
# save_labels = train_val_data_iterator.save_val_images(exp_config.DATASET_PATH)

initialize_model_train_and_get_features(experiment_name=experiment_name,
                                        z_dim=z_dim,
                                        run_id=run_id,
                                        create_split=create_split,
                                        num_epochs=num_epochs,
                                        num_cluster_config=num_cluster_config,
                                        manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                        supervise_weight=0,
                                        beta=0,
                                        reconstruction_weight=1,
                                        model_type=MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,
                                        num_units=num_units,
                                        save_reconstructed_images=True,
                                        split_name="split_1",
                                        train_val_data_iterator=train_val_data_iterator,
                                        num_val_samples=128,
                                        learning_rate=0.001,
                                        dataset_name="cifar_10",
                                        activation_output_layer="LINEAR",
                                        num_decoder_layer=len(num_units) + 1,
                                        write_predictions=False,
                                        seed=547
                                        )
