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
"""
Testcase 
"""

class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        z_dim = 32
        num_units = [64, 128, 64, 64]
        # num_units = [128, 256, 512, 1024]
        learning_rate = 1e-3
        num_epochs = 100
        create_split = True
        run_id = 7
        num_cluster_config = ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW
        beta = 0
        supervise_weight = 0
        reconstruction_weight = 1
        dataset_name = "cifar_10"
        split_name = "split_1"
        num_val_samples = 128
        dao = get_dao(dataset_name, split_name, num_val_samples, num_training_samples=512)
        exp_config = ExperimentConfig(root_path=root_path,
                                      num_decoder_layer=len(num_units)+1,
                                      z_dim=z_dim,
                                      num_units=num_units,
                                      num_cluster_config=num_cluster_config,
                                      confidence_decay_factor=5,
                                      beta=beta,
                                      supervise_weight=supervise_weight,
                                      dataset_name=dataset_name,
                                      split_name=split_name,
                                      model_name="VAE",
                                      batch_size=64,
                                      eval_interval_in_epochs=1,
                                      name=experiment_name,
                                      num_val_samples=num_val_samples,
                                      total_training_samples=dao.number_of_training_samples,
                                      manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                      reconstruction_weight=reconstruction_weight,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer="LINEAR",
                                      save_reconstructed_images=True,
                                      learning_rate=learning_rate,
                                      run_evaluation_during_training=True,
                                      write_predictions=True,
                                      model_save_interval=1,
                                      seed=547
                                      )
        exp_config.check_and_create_directories(run_id, create=True)
        exp = Experiment(run_id, experiment_name, exp_config, run_id)
        print(exp.as_json())
        with open(exp_config.BASE_PATH + "config.json", "w") as config_file:
            json.dump(exp_config.as_json(), config_file)
        num_epochs_completed = 0

        train_val_data_iterator = get_train_val_iterator(create_split,
                                                         dao,
                                                         exp_config,
                                                         num_epochs_completed,
                                                         split_name)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
            model = get_model(dao=dao,
                              exp_config=exp_config,
                              model_type=MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
                              num_epochs=2,
                              sess=sess,
                              test_data_iterator=None,
                              train_val_data_iterator=train_val_data_iterator
                              )

            exp_config, predicted_df = train_and_get_features(exp, model, train_val_data_iterator,)

        self.assertIsNotNone(exp_config)
        self.assertIsNotNone(predicted_df)


if __name__ == '__main__':
    unittest.main()
