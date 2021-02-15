import unittest
import tensorflow as tf

from clearn.config import ExperimentConfig
from clearn.dao.dao_factory import get_dao
from clearn.experiments.experiment import MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10, load_trained_model, \
    load_model_and_test

root_path = "/Users/sunilv/concept_learning_exp"
experiment_name = "Experiment_5"


"""
Test case flow 
1. Load trained mode 
2. Evaluate  - Passing

"""
class TestLoadTrainedModel(unittest.TestCase):
    # def test_load_trained_model(self):
    #     z_dim = 32
    #     num_units = [64, 128, 64, 64]
    #     # num_units = [128, 256, 512, 1024]
    #     learning_rate = 1e-3
    #     num_epochs = 100
    #     num_runs = 1
    #     create_split = True
    #     run_id = 2
    #     num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
    #     train_val_data_iterator = None
    #     beta = 0
    #     supervise_weight = 0
    #     reconstruction_weight = 1
    #     dataset_name = "cifar_10"
    #     split_name = "split_1"
    #     num_val_samples = 5000
    #     dao = get_dao(dataset_name, split_name, num_val_samples)
    #     tf.reset_default_graph()
    #
    #     model, exp_config, _, _, num_training_epochs_completed = load_trained_model(experiment_name=experiment_name,
    #                                                                                 root_path=root_path,
    #                                                                                 z_dim=z_dim,
    #                                                                                 run_id=run_id,
    #                                                                                 num_cluster_config=num_cluster_config,
    #                                                                                 manual_labels_config=ExperimentConfig.USE_ACTUAL,
    #                                                                                 supervise_weight=supervise_weight,
    #                                                                                 beta=beta,
    #                                                                                 reconstruction_weight=reconstruction_weight,
    #                                                                                 model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
    #                                                                                 num_units=num_units,
    #                                                                                 save_reconstructed_images=True,
    #                                                                                 split_name=split_name,
    #                                                                                 num_val_samples=num_val_samples,
    #                                                                                 learning_rate=learning_rate,
    #                                                                                 dataset_name=dataset_name,
    #                                                                                 activation_output_layer="LINEAR",
    #                                                                                 write_predictions=True,
    #                                                                                 seed=547,
    #                                                                                 eval_interval_in_epochs=5,
    #                                                                                 )
    #     self.assertEqual(num_training_epochs_completed, 96)

    def test_evaluate(self):
        z_dim = 32
        num_units = [64, 128, 64, 64]
        # num_units = [128, 256, 512, 1024]
        learning_rate = 1e-3
        num_epochs = 100
        num_runs = 1
        create_split = True
        run_id = 2
        num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
        train_val_data_iterator = None
        beta = 0
        supervise_weight = 0
        reconstruction_weight = 1
        dataset_name = "cifar_10"
        split_name = "split_1"
        num_val_samples = 5000
        dao = get_dao(dataset_name, split_name, num_val_samples)
        tf.reset_default_graph()

        exp_config, predicted_df = load_model_and_test(experiment_name=experiment_name,
                                                       root_path=root_path,
                                                       z_dim=z_dim,
                                                       run_id=run_id,
                                                       num_cluster_config=num_cluster_config,
                                                       model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
                                                       num_units=num_units,
                                                       save_reconstructed_images=True,
                                                       split_name=split_name,
                                                       num_val_samples=num_val_samples,
                                                       dataset_name=dataset_name,
                                                       write_predictions=True,
                                                       )
        self.assertIsNotNone(exp_config)
        self.assertIsNotNone(predicted_df)


if __name__ == '__main__':
    unittest.main()
