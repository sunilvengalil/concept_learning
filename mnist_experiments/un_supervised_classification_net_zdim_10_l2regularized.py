
import json
import os
import tensorflow as tf

from clearn.experiments.experiment import Experiment, train_and_get_features
from clearn.models.classify.classifier import ClassifierModel
from clearn.utils.data_loader import TrainValDataIterator
from clearn.config import ExperimentConfig

experiment_name = "unsupervised_vae_z_dim_10_l2_regularized"
num_epochs = 10
create_split = True
z_dim = 10
run_id = 0
exp_config = ExperimentConfig(root_path="/Users/sunilv/concept_learning_exp",
                              num_decoder_layer=4,
                              z_dim=z_dim,
                              num_units=[64, 128, 32],
                              num_cluster_config=None,
                              confidence_decay_factor=5,
                              beta=5,
                              supervise_weight=150,
                              dataset_name="mnist",
                              split_name="Split_1",
                              model_name="VAE",
                              batch_size=64,
                              eval_interval=300,
                              name=experiment_name,
                              num_val_samples=128,
                              total_training_samples=60000,
                              manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                              reconstruction_weight=1,
                              activation_hidden_layer="RELU",
                              activation_output_layer="SIGMOID"
                              )

exp_config.check_and_create_directories(run_id, create=True)

# TODO make this a configuration
# To change output type from sigmoid to leaky relu, do the following
# 1. In vae.py change the output layer type in decode()
# 2. Change the loss function in build_model
exp = Experiment(1, "VAE_MNIST", exp_config, run_id)
print(exp.as_json())
with open(exp_config.BASE_PATH + "config.json", "w") as config_file:
    json.dump(exp_config.as_json(), config_file)
if create_split:
    train_val_data_iterator = TrainValDataIterator(exp.config.DATASET_ROOT_PATH,
                                                   shuffle=True,
                                                   stratified=True,
                                                   validation_samples=exp.config.num_val_samples,
                                                   split_names=["train", "validation"],
                                                   split_location=exp.config.DATASET_PATH,
                                                   batch_size=exp.config.BATCH_SIZE)
else:
    manual_annotation_file = os.path.join(exp_config.ANALYSIS_PATH,
                                          f"manual_annotation_epoch_{num_epochs - 1:.1f}.csv"
                                          )

    train_val_data_iterator = TrainValDataIterator.from_existing_split(exp.config.split_name,
                                                                       exp.config.DATASET_PATH,
                                                                       exp.config.BATCH_SIZE,
                                                                       manual_labels_config=exp.config.manual_labels_config,
                                                                       manual_annotation_file=None)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    model = ClassifierModel(exp_config=exp_config,
                            sess=sess,
                            epoch=num_epochs,
                            batch_size=exp_config.BATCH_SIZE,
                            z_dim=exp_config.Z_DIM,
                            dataset_name=exp_config.dataset_name,
                            beta=exp_config.beta,
                            num_units_in_layer=exp_config.num_units,
                            train_val_data_iterator=train_val_data_iterator,
                            log_dir=exp.config.LOG_PATH,
                            checkpoint_dir=exp.config.TRAINED_MODELS_PATH,
                            result_dir=exp.config.PREDICTION_RESULTS_PATH,
                            supervise_weight=exp.config.supervise_weight,
                            reconstruction_weight=exp.config.reconstruction_weight,
                            reconstructed_image_dir=exp.config.reconstructed_images_path
                            )
    train_and_get_features(exp, model, train_val_data_iterator, num_epochs)
