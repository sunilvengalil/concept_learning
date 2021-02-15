import json
import unittest
import tensorflow as tf

from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10, train_and_get_features, \
    get_train_val_iterator, get_model
from clearn.experiments.experiment import Experiment

root_path = "/Users/sunilv/concept_learning_exp"
experiment_name = "Experiment_5"


class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        from clearn.config import ExperimentConfig
        from clearn.dao.dao_factory import get_dao
        from clearn.experiments.experiment import initialize_model_train_and_get_features, \
            MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10

        experiment_name = "Experiment_5"
        root_path = "/Users/sunilv/concept_learning_exp/"
        z_dim = 32
        num_units = [64, 128, 64, 64]
        # num_units = [128, 256, 512, 1024]
        learning_rate = 1e-4
        num_epochs = 5
        num_runs = 1
        create_split = True
        run_id = 1
        num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
        train_val_data_iterator = None
        beta = 0
        supervise_weight = 0
        reconstruction_weight = 1
        dataset_name = "cifar_10"
        split_name = "split_1"
        num_val_samples = 5000
        dao = get_dao(dataset_name, split_name, num_val_samples, 512)
        initialize_model_train_and_get_features(experiment_name=experiment_name,
                                                z_dim=z_dim,
                                                run_id=run_id,
                                                create_split=create_split,
                                                num_epochs=num_epochs,
                                                num_cluster_config=num_cluster_config,
                                                manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                                supervise_weight=supervise_weight,
                                                beta=beta,
                                                reconstruction_weight=reconstruction_weight,
                                                model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
                                                num_units=num_units,
                                                save_reconstructed_images=True,
                                                split_name=split_name,
                                                train_val_data_iterator=train_val_data_iterator,
                                                num_val_samples=num_val_samples,
                                                learning_rate=learning_rate,
                                                dataset_name=dataset_name,
                                                activation_output_layer="LINEAR",
                                                num_decoder_layer=len(num_units) + 1,
                                                write_predictions=True,
                                                seed=547,
                                                model_save_interval=5,
                                                eval_interval_in_epochs=5,
                                                dao=dao
                                                )


if __name__ == '__main__':
    unittest.main()
