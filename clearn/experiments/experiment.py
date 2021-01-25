import tensorflow as tf
from clearn.analysis.encode_images import encode_images, encode_images_and_get_features
import json
import os

from clearn.models.classify.Cifar10Classifier import Cifar10Classifier
from clearn.models.classify.classifier import ClassifierModel
from clearn.models.classify.supervised_classifier import SupervisedClassifierModel
from clearn.dao.dao_factory import get_dao

from clearn.utils.data_loader import TrainValDataIterator
from clearn.config import ExperimentConfig
from clearn.utils.utils import show_all_variables

MODEL_TYPE_SEMI_SUPERVISED_CLASSIFIER = "classifier"
MODEL_TYPE_SUPERVISED_CLASSIFIER = "supervised_classifier"


class Experiment:

    def __init__(self, exp_id, name, config1: ExperimentConfig, _run_id=None):
        """
        :type config1: ExperimentConfig
        """
        if _run_id is None:
            self.run_id = id
        else:
            self.run_id = _run_id
        self.id = exp_id
        self.name = name
        self.config = config1
        self.model = None

    def initialize(self, _model=None):
        """

        :type _model: VAE
        """
        self.model = _model
        self.config.check_and_create_directories(self.run_id)

    def as_json(self):
        config_json = self.config.as_json()
        config_json["RUN_ID"] = self.run_id
        config_json["ID"] = self.id
        config_json["name"] = self.name
        return config_json

    def train(self, train_val_data_iterator=None, _create_split=False):
        if train_val_data_iterator is None:

            if _create_split:
                train_val_data_iterator = TrainValDataIterator(self.config.DATASET_ROOT_PATH,
                                                               shuffle=True,
                                                               stratified=True,
                                                               validation_samples=self.config.num_val_samples,
                                                               split_names=["train", "validation"],
                                                               split_location=self.config.DATASET_PATH,
                                                               batch_size=self.config.BATCH_SIZE)
            else:
                train_val_data_iterator = TrainValDataIterator.from_existing_split(self.config.split_name,
                                                                                   self.config.DATASET_PATH,
                                                                                   self.config.BATCH_SIZE,
                                                                                   manual_labels_config=self.config.manual_labels_config)
        self.model.train(train_val_data_iterator)

        print(" [*] Training finished!")

    def encode_latent_vector(self, _train_val_data_iterator, epoch, dataset_type,
                             save_results=True):
        return encode_images(self.model,
                             _train_val_data_iterator,
                             self.config,
                             epoch,
                             dataset_type,
                             save_results
                             )

    def encode_latent_vector_and_get_features(self, _train_val_data_iterator, epoch, dataset_type,
                                              save_results=True):
        return encode_images_and_get_features(self.model,
                                              _train_val_data_iterator,
                                              self.config,
                                              epoch,
                                              dataset_type,
                                              save_results
                                              )


def load_trained_model(experiment_name,
                       z_dim,
                       run_id,
                       num_cluster_config=None,
                       manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                       supervise_weight=150,
                       beta=5,
                       reconstruction_weight=1,
                       model_type="classifier"
                       ):
    exp_config = ExperimentConfig(root_path="/Users/sunilv/concept_learning_exp",
                                  num_decoder_layer=4,
                                  z_dim=z_dim,
                                  num_units=[64, 128, 32],
                                  num_cluster_config=num_cluster_config,
                                  confidence_decay_factor=5,
                                  beta=beta,
                                  supervise_weight=supervise_weight,
                                  dataset_name="mnist",
                                  split_name="Split_1",
                                  model_name="VAE",
                                  batch_size=64,
                                  eval_interval=300,
                                  name=experiment_name,
                                  num_val_samples=128,
                                  total_training_samples=60000,
                                  manual_labels_config=manual_labels_config,
                                  reconstruction_weight=reconstruction_weight,
                                  activation_hidden_layer="RELU",
                                  activation_output_layer="SIGMOID"
                                  )
    exp_config.check_and_create_directories(run_id)
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if model_type == MODEL_TYPE_SEMI_SUPERVISED_CLASSIFIER:
            model = ClassifierModel(exp_config,
                                    sess,
                                    epoch=1,
                                    batch_size=exp_config.BATCH_SIZE,
                                    z_dim=z_dim,
                                    dataset_name=exp_config.dataset_name,
                                    beta=exp_config.beta,
                                    num_units_in_layer=exp_config.num_units,
                                    log_dir=exp_config.LOG_PATH,
                                    checkpoint_dir=exp_config.TRAINED_MODELS_PATH,
                                    result_dir=exp_config.PREDICTION_RESULTS_PATH
                                    )
            print(model.get_trainable_vars())
            num_steps_completed = model.counter
            print("Number of steps completed={}".format(num_steps_completed))
            num_batches = exp_config.num_train_samples / exp_config.BATCH_SIZE
            epochs_completed = num_steps_completed // num_batches
            print("Number of epochs completed {}".format(epochs_completed))
        elif model_type == MODEL_TYPE_SUPERVISED_CLASSIFIER:
            model = SupervisedClassifierModel(exp_config,
                                              sess,
                                              epoch=1,
                                              batch_size=exp_config.BATCH_SIZE,
                                              z_dim=z_dim,
                                              dataset_name=exp_config.dataset_name,
                                              beta=exp_config.beta,
                                              num_units_in_layer=exp_config.num_units,
                                              log_dir=exp_config.LOG_PATH,
                                              checkpoint_dir=exp_config.TRAINED_MODELS_PATH,
                                              result_dir=exp_config.PREDICTION_RESULTS_PATH
                                              )
            print(model.get_trainable_vars())
            num_steps_completed = model.counter
            print("Number of steps completed={}".format(num_steps_completed))
            num_batches = exp_config.num_train_samples / exp_config.BATCH_SIZE
            epochs_completed = num_steps_completed // num_batches
            print("Number of epochs completed {}".format(epochs_completed))
        else:
            raise Exception(
                f"model_type should be one of [{MODEL_TYPE_SEMI_SUPERVISED_CLASSIFIER}, {MODEL_TYPE_SUPERVISED_CLASSIFIER}]")
        return model, exp_config, model.get_encoder_weights_bias(), model.get_decoder_weights_bias(), epochs_completed


def initialize_model_train_and_get_features(experiment_name,
                                            z_dim,
                                            run_id,
                                            create_split,
                                            num_epochs,
                                            num_cluster_config=None,
                                            manual_annotation_file=None,
                                            manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                            supervise_weight=150,
                                            beta=5,
                                            reconstruction_weight=1,
                                            model_type="classifier",
                                            num_units=[64, 128, 32],
                                            save_reconstructed_images=True,
                                            split_name="Split_1",
                                            train_val_data_iterator=None,
                                            num_val_samples=128,
                                            root_path="/Users/sunilv/concept_learning_exp",
                                            learning_rate=0.001,
                                            run_evaluation_during_training=True,
                                            eval_interval=300,
                                            dataset_name="mnist",
                                            activation_output_layer="SIGMOID",
                                            write_predictions=True,
                                            num_decoder_layer =4):
    dao = get_dao(dataset_name, split_name)
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
                                  eval_interval=eval_interval,
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
    exp = Experiment(1,experiment_name, exp_config, run_id)
    print(exp.as_json())
    with open(exp_config.BASE_PATH + "config.json", "w") as config_file:
        json.dump(exp_config.as_json(), config_file)
    if train_val_data_iterator is None:
        split_filename = exp.config.DATASET_PATH + split_name + ".json"
        manual_annotation_file = os.path.join(exp_config.ANALYSIS_PATH,
                                              f"manual_annotation_epoch_{num_epochs - 1:.1f}.csv"
                                              )
        print(split_filename)
        if os.path.isfile(split_filename):
            if manual_annotation_file is not None:
                train_val_data_iterator = TrainValDataIterator.from_existing_split(exp.config.split_name,
                                                                                   exp.config.DATASET_PATH,
                                                                                   exp.config.BATCH_SIZE,
                                                                                   manual_labels_config=exp.config.manual_labels_config,
                                                                                   manual_annotation_file=manual_annotation_file,
                                                                                   dao=dao)
        elif create_split:
            train_val_data_iterator = TrainValDataIterator(exp.config.DATASET_ROOT_PATH,
                                                           shuffle=True,
                                                           stratified=True,
                                                           validation_samples=exp.config.num_val_samples,
                                                           split_names=["train", "validation"],
                                                           split_location=exp.config.DATASET_PATH,
                                                           batch_size=exp.config.BATCH_SIZE,
                                                           manual_labels_config=exp.config.manual_labels_config,
                                                           manual_annotation_file=manual_annotation_file,
                                                           dao=dao)
        else:
            raise Exception(f"File does not exists {split_filename}")

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        if model_type == MODEL_TYPE_SEMI_SUPERVISED_CLASSIFIER:
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
                                    reconstructed_image_dir=exp.config.reconstructed_images_path,
                                    run_evaluation_during_training=run_evaluation_during_training,
                                    dao=dao,
                                    write_predictions=write_predictions
                                    )
        elif model_type == MODEL_TYPE_SUPERVISED_CLASSIFIER:
            model = SupervisedClassifierModel(exp_config=exp_config,
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
                                              reconstructed_image_dir=exp.config.reconstructed_images_path,
                                              dao=dao,
                                              write_predictions=write_predictions
                                              )
        elif model_type == "cifar_arch_vaal":
            model = Cifar10Classifier(exp_config=exp_config,
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
                                              reconstructed_image_dir=exp.config.reconstructed_images_path,
                                              dao=dao,
                                              write_predictions=write_predictions
                                              )
        else:
            raise Exception(
                f"model_type should be one of [{MODEL_TYPE_SEMI_SUPERVISED_CLASSIFIER}, {MODEL_TYPE_SUPERVISED_CLASSIFIER}]")
        print("Starting training")
        train_and_get_features(exp, model, train_val_data_iterator, num_epochs)
        train_val_data_iterator.reset_counter("train")
        train_val_data_iterator.reset_counter("val")
        return train_val_data_iterator, exp_config, model


def train_and_get_features(exp: Experiment,
                           model: ClassifierModel,
                           train_val_data_iterator: TrainValDataIterator,
                           num_epochs):
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
