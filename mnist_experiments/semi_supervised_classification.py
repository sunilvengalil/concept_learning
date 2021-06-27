import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST
from clearn.config import ExperimentConfig

create_split = False
z_dim = 10
experiment_name = "semi_supervised_classification_test"


if __name__ == '__main__':
    # parse arguments
    num_epochs = 2
    num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
    manual_annotation_file = f"manual_annotation_epoch_{num_epochs - 1:.1f}.csv"
    run_id = 1000
    initialize_model_train_and_get_features(experiment_name=experiment_name,
                                            z_dim=z_dim,
                                            batch_size=64,
                                            run_id=run_id,
                                            create_split=create_split,
                                            num_epochs=num_epochs,
                                            num_cluster_config=num_cluster_config,
                                            num_units=[4, 4],
                                            eval_interval_in_epochs=1,
                                            model_save_interval=1,
                                            model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST,
                                            strides=[2, 1, 1],
                                            num_dense_layers=1,
                                            translate_image=True
                                            )
    tf.reset_default_graph()
