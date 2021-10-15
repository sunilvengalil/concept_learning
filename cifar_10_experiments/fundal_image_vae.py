import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_UNSUPERVISED
from clearn.config import ExperimentConfig


if __name__ == '__main__':
    # parse arguments
    num_epochs = 20
    create_split = True
    z_dim = 64
    experiment_name = "Experiment_4"
    num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
    run_id = 201
    train_val_data_iterator = None
    tf.reset_default_graph()

num_units = [1024, 512, 256, 128, 64, 32]
train_val_data_iterator, exp_config, model = initialize_model_train_and_get_features(root_path="C:/concept_learning_exp/",
                                                                                     experiment_name=experiment_name,
                                                                                     z_dim=z_dim,
                                                                                     run_id=run_id,
                                                                                     create_split=create_split,
                                                                                     num_epochs=num_epochs,
                                                                                     num_cluster_config=num_cluster_config,
    manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
    supervise_weight=0,
    beta=0,
    reconstruction_weight=1,
    num_units=num_units,
    save_reconstructed_images=True,
    split_name="split_1",
    train_val_data_iterator=train_val_data_iterator,
    num_val_samples=0,
    learning_rate=0.001,
    dataset_name="drive_processed",
    activation_output_layer="LINEAR",
    write_predictions=False,
    model_save_interval= 5,
    num_decoder_layer=5,
    model_type=MODEL_TYPE_VAE_UNSUPERVISED,
    num_dense_layers=0,
    strides=[2, 2, 2, 2, 2, 2, 1],
    run_test=False,
    fully_convolutional=True
    )
