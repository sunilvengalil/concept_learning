from clearn.config import ExperimentConfig
from clearn.dao.cifar_10 import CiFar10Dao

from clearn.utils.data_loader import TrainValDataIterator, DataIterator


def get_data_iterators(exp_config, dao):
    # train_val_data_iterator = TrainValDataIterator(exp_config.DATASET_ROOT_PATH,
    #                                                shuffle=True,
    #                                                stratified=True,
    #                                                validation_samples=exp_config.num_val_samples,
    #                                                split_names=["train", "validation"],
    #                                                split_location=exp_config.DATASET_PATH,
    #                                                batch_size=exp_config.BATCH_SIZE,
    #                                                manual_labels_config=exp_config.manual_labels_config,
    #                                                manual_annotation_file=None,
    #                                                dao=dao)
    # test_data_iterator = DataIterator(exp_config.DATASET_ROOT_PATH,
    #                                   exp_config.DATASET_PATH,
    #                                   ["test"],
    #                                   exp_config.BATCH_SIZE,
    #                                   dao=dao)

    train_val_data_iterator = TrainValDataIterator.from_existing_split(exp_config.split_name,
                                                                        exp_config.DATASET_PATH,
                                                                        exp_config.BATCH_SIZE,
                                                                        manual_labels_config=exp_config.manual_labels_config,
                                                                        manual_annotation_file=manual_annotation_file,
                                                                        dao=dao)

    test_data_iterator = DataIterator.from_existing_split("test",
                                                          split_location=exp_config.DATASET_ROOT_PATH + "/test/",
                                                          batch_size=exp_config.BATCH_SIZE,
                                                          dao=dao
                                                          )

    return train_val_data_iterator, test_data_iterator



create_split = False
z_dim = 32
experiment_name = "cifar_arch_vaal_split_1"

num_epochs = 30
num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
run_id = 5
train_val_data_iterator = None
num_units = [32,64, 64, 64, 64]

manual_annotation_file=None
manual_labels_config=ExperimentConfig.USE_ACTUAL
supervise_weight=1
beta=0
reconstruction_weight=0
model_type="cifar_arch_vaal"
num_units=num_units
save_reconstructed_images=False
split_name="split_1"
num_val_samples=5000
learning_rate=0.001
dataset_name="cifar_10"
activation_output_layer="LINEAR"
write_predictions=False
num_decoder_layer=6
root_path = "/Users/sunilv/concept_learning_exp"

dao = CiFar10Dao(split_name)


exp_config = ExperimentConfig(root_path=root_path,
                              num_decoder_layer=num_decoder_layer,
                              z_dim=z_dim,
                              num_units=num_units,
                              num_cluster_config=num_cluster_config,
                              confidence_decay_factor=5,
                              beta=beta,
                              supervise_weight=supervise_weight,
                              dataset_name=dataset_name,
                              split_name=split_name,
                              model_name="VAE",
                              batch_size=128,
                              eval_interval=300,
                              name=experiment_name,
                              num_val_samples=num_val_samples,
                              total_training_samples=dao.number_of_training_samples,
                              manual_labels_config=manual_labels_config,
                              reconstruction_weight=reconstruction_weight,
                              activation_hidden_layer="RELU",
                              activation_output_layer=activation_output_layer,
                              save_reconstructed_images=save_reconstructed_images,
                              learning_rate=learning_rate
                              )
exp_config.check_and_create_directories(run_id, create=True)

train_val_data_iterator, test_data_iterator = get_data_iterators(exp_config, dao)
