
import json
import os
import tensorflow as tf

from clearn.experiments.experiment import Experiment
from clearn.models.classify.classifier import ClassifierModel
from clearn.utils.data_loader import TrainValDataIterator
from clearn.config import ExperimentConfig
from clearn.utils.utils import show_all_variables

experiment_name = "semi_supervised_classification"
z_dim_range = [1, 22, 4]
num_epochs = 10
num_runs = 5
create_split = False
completed_z_dims = 0
for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
    if z_dim < completed_z_dims:
        continue
    completed_runs = 0
    for run_id in range(num_runs):
        if run_id < completed_runs:
            continue
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
                                      name="semi_supervised_classification",
                                      num_val_samples=128,
                                      total_training_samples=60000,
                                      manual_labels_config=TrainValDataIterator.USE_CLUSTER_CENTER,
                                      reconstruction_weight=1,
                                      activation_hidden_layer="RELU",
                                      activation_output_layer="SIGMOID"
                                      )
        BATCH_SIZE = exp_config.BATCH_SIZE
        DATASET_NAME = exp_config.dataset_name
        exp_config.check_and_create_directories(run_id, create=True)

        # TODO make this a configuration
        # To change output type from sigmoid to leaky relu, do the following
        # 1. In vae.py change the output layer type in decode()
        # 2. Change the loss function in build_model
        exp = Experiment(1, "VAE_MNIST", 128, exp_config, run_id)
        print(exp.as_json())
        with open(exp_config.BASE_PATH + "config.json", "w") as config_file:
            json.dump(exp_config.as_json(), config_file)
        if create_split:
            train_val_data_iterator = TrainValDataIterator(exp.config.DATASET_ROOT_PATH,
                                                           shuffle=True,
                                                           stratified=True,
                                                           validation_samples=exp.num_validation_samples,
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
                                    dataset_name=DATASET_NAME,
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
            exp.model = model
            # show network architecture
            show_all_variables()
            exp.train(train_val_data_iterator)

            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            exp.encode_latent_vector(train_val_data_iterator, num_epochs, "train")

            train_val_data_iterator.reset_counter("train")
            train_val_data_iterator.reset_counter("val")
            exp.encode_latent_vector(train_val_data_iterator, num_epochs, "val")
        tf.reset_default_graph()
        completed_runs = completed_runs + 1
