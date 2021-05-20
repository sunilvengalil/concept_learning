import logging

import tensorflow as tf
from clearn.analysis.encode_images import encode_images, encode_images_and_get_features
import json
import os

from clearn.dao.idao import IDao
from clearn.models.classify.cifar_10_classifier import Cifar10Classifier, Cifar10F
from clearn.models.classify.cifar_10_vae import Cifar10Vae
from clearn.models.classify.semi_supervised import SemiSupervisedClassifier
from clearn.models.classify.semi_supervised_mnist import SemiSupervisedClassifierMnist
from clearn.models.segment.semisupervised_segmentation_mnist import SemiSupervisedSegmenterMnist
from clearn.models.vae import VAE
from clearn.models.classify.supervised_classifier import SupervisedClassifierModel
from clearn.dao.dao_factory import get_dao
from clearn.models.model import Model

from clearn.utils.data_loader import TrainValDataIterator, DataIterator
from clearn.config import ExperimentConfig
from clearn.utils.utils import show_all_variables, get_padding_info

MODEL_TYPE_VAE_UNSUPERVISED = "VAE"
MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST = "VAE_SEMI_SUPERVISED_MNIST"
MODEL_TYPE_SUPERVISED_CLASSIFIER = "CLASSIFIER_SUPERVISED"

MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10 = "VAE_UNSUPERVISED_CIFAR10"
VAAL_ARCHITECTURE_FOR_CIFAR = "ACTIVE_LEARNING_VAAL_CIFAR"
CIFAR_VGG = "CIFAR_VGG"
CIFAR10_F = "CIFAR10_F"
MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10 = "VAE_SEMI_SUPERVISED_CIFAR10"
VAE_FCNN = "VAE_FCNN"

model_types = [MODEL_TYPE_VAE_UNSUPERVISED,
               MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,
               MODEL_TYPE_SUPERVISED_CLASSIFIER,
               VAAL_ARCHITECTURE_FOR_CIFAR,
               CIFAR_VGG,
               CIFAR10_F,
               MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10,
               VAE_FCNN]


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
                train_val_data_iterator = TrainValDataIterator(dataset_path=self.config.DATASET_ROOT_PATH,
                                                               dao=self.model.dao,
                                                               shuffle=True,
                                                               stratified=True,
                                                               validation_samples=self.config.num_val_samples,
                                                               split_names=["train", "validation"],
                                                               split_location=self.config.DATASET_PATH,
                                                               batch_size=self.config.BATCH_SIZE)
            else:
                train_val_data_iterator = TrainValDataIterator.from_existing_split(split_name=self.config.split_name,
                                                                                   split_location=self.config.DATASET_PATH,
                                                                                   batch_size=self.config.BATCH_SIZE,
                                                                                   manual_labels_config=self.config.manual_labels_config,
                                                                                   dao=self.model.dao)
        self.model.train(train_val_data_iterator)

        print(" [*] Training finished!")

    def test(self, data_iterator):
        return self.model.evaluate(data_iterator, dataset_type="test")

    def encode_latent_vector(self, _train_val_data_iterator, dataset_type,
                             save_images=True):
        encoded_df = encode_images(self.model,
                                   _train_val_data_iterator,
                                   dataset_type=dataset_type,
                                   save_images=save_images
                                   )
        return encoded_df, self.model.exp_config

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
                       root_path,
                       num_units,
                       dataset_name,
                       split_name,
                       num_val_samples,
                       activation_output_layer,
                       save_reconstructed_images,
                       learning_rate,
                       write_predictions,
                       seed,
                       z_dim,
                       run_id,
                       model_type,
                       num_dense_layers,
                       strides,
                       num_cluster_config=None,
                       manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                       supervise_weight=150,
                       beta=5,
                       reconstruction_weight=1,
                       eval_interval_in_epochs=1

                       ):
    dao = get_dao(dataset_name, split_name, num_val_samples)

    exp_config = ExperimentConfig(root_path=root_path,
                                  num_decoder_layer=len(num_units) + 1,
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
                                  name=experiment_name,
                                  num_val_samples=num_val_samples,
                                  total_training_samples=dao.number_of_training_samples,
                                  manual_labels_config=manual_labels_config,
                                  reconstruction_weight=reconstruction_weight,
                                  activation_hidden_layer="RELU",
                                  activation_output_layer=activation_output_layer,
                                  save_reconstructed_images=save_reconstructed_images,
                                  learning_rate=learning_rate,
                                  write_predictions=write_predictions,
                                  eval_interval_in_epochs=eval_interval_in_epochs,
                                  seed=seed,
                                  strides=strides,
                                  num_dense_layers=num_dense_layers
                                  )
    exp_config.check_and_create_directories(run_id, create=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = get_model(dao, exp_config, model_type, num_epochs=0, sess=sess, test_data_iterator=None,
                          train_val_data_iterator=None)
        return model, exp_config, model.get_encoder_weights_bias(), model.get_decoder_weights_bias(), model.num_training_epochs_completed


def initialize_model_train_and_get_features(experiment_name,
                                            z_dim,
                                            run_id,
                                            create_split,
                                            num_epochs,
                                            model_type,
                                            strides,
                                            num_dense_layers,
                                            num_cluster_config=None,
                                            manual_labels_config=ExperimentConfig.USE_CLUSTER_CENTER,
                                            supervise_weight=150,
                                            beta=5,
                                            reconstruction_weight=1,
                                            num_units=None,
                                            save_reconstructed_images=True,
                                            split_name="Split_1",
                                            train_val_data_iterator=None,
                                            num_val_samples=128,
                                            root_path="/Users/sunilv/concept_learning_exp",
                                            learning_rate=0.001,
                                            run_evaluation_during_training=True,
                                            eval_interval_in_epochs=1,
                                            dataset_name="mnist",
                                            activation_output_layer="SIGMOID",
                                            write_predictions=True,
                                            num_decoder_layer=4,
                                            test_data_iterator=None,
                                            seed=547,
                                            num_epochs_completed=0,
                                            model_save_interval=1,
                                            budget=1,
                                            confidence_decay_factor=5,
                                            dao: IDao = None,
                                            distance_metric=ExperimentConfig.DISTANCE_EUCLIDEAN,
                                            clustering_alg=ExperimentConfig.CLUSTERING_K_MEANS,
                                            confidence_decay_function=ExperimentConfig.CONFIDENCE_DECAY_FUNCTION_EXPONENTIAL,
                                            batch_size=64,
                                            return_latent_vector=True,
                                            log_level=logging.INFO,
                                            fully_convolutional = False,
                                            num_concepts=10,
                                            supervise_weight_concepts=1,
                                            num_individual_samples_annotated=0,
                                            num_samples_wrongly_annotated=0,
                                            total_confidence_of_wrong_annotation=0,
                                            uncorrelated_features=False
                                            ):
    if dao is None:
        dao = get_dao(dataset_name, split_name, num_val_samples)

    if num_units is None:
        num_units = [64, 128, 32]
    exp_config = ExperimentConfig(root_path=root_path,
                                  num_decoder_layer=num_decoder_layer,
                                  z_dim=z_dim,
                                  num_units=num_units,
                                  num_dense_layers=num_dense_layers,
                                  num_cluster_config=num_cluster_config,
                                  confidence_decay_factor=confidence_decay_factor,
                                  beta=beta,
                                  supervise_weight=supervise_weight,
                                  dataset_name=dataset_name,
                                  split_name=split_name,
                                  model_name="VAE",
                                  batch_size=batch_size,
                                  eval_interval_in_epochs=eval_interval_in_epochs,
                                  name=experiment_name,
                                  num_val_samples=num_val_samples,
                                  total_training_samples=dao.number_of_training_samples,
                                  manual_labels_config=manual_labels_config,
                                  reconstruction_weight=reconstruction_weight,
                                  activation_hidden_layer="RELU",
                                  activation_output_layer=activation_output_layer,
                                  save_reconstructed_images=save_reconstructed_images,
                                  learning_rate=learning_rate,
                                  run_evaluation_during_training=run_evaluation_during_training,
                                  write_predictions=write_predictions,
                                  model_save_interval=model_save_interval,
                                  seed=seed,
                                  budget=budget,
                                  confidence_decay_function=confidence_decay_function,
                                  distance_metric=distance_metric,
                                  clustering_alg=clustering_alg,
                                  return_latent_vector=return_latent_vector,
                                  log_level=log_level,
                                  fully_convolutional=fully_convolutional,
                                  num_concepts=num_concepts,
                                  supervise_weight_concepts=supervise_weight_concepts,
                                  strides=strides,
                                  uncorrelated_features=uncorrelated_features
                                  )
    exp_config.check_and_create_directories(run_id, create=True)
    exp = Experiment(1, experiment_name, exp_config, run_id)
    print(exp.as_json())
    with open(exp_config.BASE_PATH + "config.json", "w") as config_file:
        json.dump(exp_config.as_json(), config_file)
    if train_val_data_iterator is None:
        train_val_data_iterator = get_train_val_iterator(create_split,
                                                         dao,
                                                         exp_config,
                                                         num_epochs_completed,
                                                         split_name
                                                         )

    if test_data_iterator is None:
        test_data_location = exp_config.DATASET_ROOT_PATH + "/test/"
        if exp_config.fully_convolutional:
            num_concepts_per_row, num_concepts_per_col = get_num_concepts_per_image(exp_config, dao)
        else:
            num_concepts_per_row, num_concepts_per_col = 1, 1
        if not os.path.isfile(test_data_location + "test.json"):
            test_data_iterator = DataIterator(dataset_path=exp_config.DATASET_ROOT_PATH,
                                              batch_size=exp_config.BATCH_SIZE,
                                              dao=dao,
                                              num_concepts_per_image_row=num_concepts_per_row,
                                              num_concepts_per_image_col=num_concepts_per_col
                                              )
        else:
            raise Exception("Test data file does not exists")
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        model = get_model(dao,
                          exp_config,
                          model_type,
                          num_epochs,
                          sess,
                          test_data_iterator,
                          train_val_data_iterator,
                          num_individual_samples_annotated=num_individual_samples_annotated,
                          num_samples_wrongly_annotated=num_individual_samples_annotated,
                          total_confidence_of_wrong_annotation=num_individual_samples_annotated
                          )
        print("Starting training")
        train_and_get_features(exp, model, train_val_data_iterator)
        train_val_data_iterator.reset_counter("train")
        train_val_data_iterator.reset_counter("val")
        return train_val_data_iterator, exp_config, model


def get_num_concepts_per_image(exp_config, dao):

    _,  _, image_sizes = get_padding_info(exp_config, dao.image_shape)
    latent_image_dim = image_sizes[len(exp_config.num_units)]
    concepts_stride = 2

    if latent_image_dim[0] % concepts_stride == 0:
        h = latent_image_dim[0] //concepts_stride
    else:
        h = (latent_image_dim[0] // concepts_stride) + 1
    if latent_image_dim[1] % concepts_stride == 0:
        w = latent_image_dim[1] // concepts_stride
    else:
        w = (latent_image_dim[1] // concepts_stride) + 1
    return h, w


def get_train_val_iterator(create_split: bool,
                           dao: IDao,
                           exp_config: ExperimentConfig,
                           num_epochs_completed: int,
                           split_name: str):
    split_filename = exp_config.DATASET_PATH + split_name + ".json"
    manual_annotation_file_name = f"manual_annotation.csv"

    manual_annotation_file = os.path.join(exp_config.ANALYSIS_PATH,
                                          manual_annotation_file_name
                                          )
    if exp_config.fully_convolutional:
        num_concepts_per_row, num_concepts_per_col = get_num_concepts_per_image(exp_config, dao)
    else:
        num_concepts_per_row, num_concepts_per_col = 1, 1
    if os.path.isfile(split_filename):
        if manual_annotation_file is not None:
            train_val_data_iterator = TrainValDataIterator.from_existing_split(dao=dao,
                                                                               split_name=exp_config.split_name,
                                                                               split_location=exp_config.DATASET_PATH,
                                                                               batch_size=exp_config.BATCH_SIZE,
                                                                               manual_labels_config=exp_config.manual_labels_config,
                                                                               manual_annotation_file=manual_annotation_file,
                                                                               budget=exp_config.budget,
                                                                               num_concepts_per_image_row=num_concepts_per_row,
                                                                               num_concepts_per_image_col=num_concepts_per_col
                                                                               )
    elif create_split:
        train_val_data_iterator = TrainValDataIterator(dataset_path=exp_config.DATASET_ROOT_PATH,
                                                       dao=dao,
                                                       shuffle=True,
                                                       stratified=True,
                                                       validation_samples=exp_config.num_val_samples,
                                                       split_names=["train", "validation"],
                                                       split_location=exp_config.DATASET_PATH,
                                                       batch_size=exp_config.BATCH_SIZE,
                                                       manual_labels_config=exp_config.manual_labels_config,
                                                       manual_annotation_file=manual_annotation_file,
                                                       seed=exp_config.seed,
                                                       budget=exp_config.budget,
                                                       num_concepts_per_image_row=num_concepts_per_row,
                                                       num_concepts_per_image_col=num_concepts_per_col
                                                       )
    else:
        raise Exception(f"File does not exists {split_filename}")
    return train_val_data_iterator


def get_model(dao: IDao,
              exp_config: ExperimentConfig,
              model_type,
              num_epochs,
              sess,
              test_data_iterator=None,
              train_val_data_iterator=None,
              check_point_epochs=None,
              num_individual_samples_annotated=0,
              num_samples_wrongly_annotated=0,
              total_confidence_of_wrong_annotation=0
              ):
    if model_type == MODEL_TYPE_SUPERVISED_CLASSIFIER:
        model = SupervisedClassifierModel(exp_config=exp_config,
                                          sess=sess,
                                          epoch=num_epochs,
                                          num_units_in_layer=exp_config.num_units,
                                          dao=dao,
                                          test_data_iterator=test_data_iterator,
                                          check_point_epochs=check_point_epochs
                                          )
    elif model_type == VAAL_ARCHITECTURE_FOR_CIFAR:
        model = Cifar10Classifier(exp_config=exp_config,
                                  sess=sess,
                                  epoch=num_epochs,
                                  num_units_in_layer=exp_config.num_units,
                                  dao=dao,
                                  test_data_iterator=test_data_iterator,
                                  check_point_epochs=check_point_epochs
                                  )
    elif model_type == CIFAR10_F:
        model = Cifar10F(exp_config=exp_config,
                         sess=sess,
                         epoch=num_epochs,
                         num_units_in_layer=exp_config.num_units,
                         dao=dao,
                         test_data_iterator=test_data_iterator,
                         check_point_epochs=check_point_epochs
                         )

    elif model_type == MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10:
        model = Cifar10Vae(exp_config=exp_config,
                           sess=sess,
                           epoch=num_epochs,
                           dao=dao,
                           check_point_epochs=check_point_epochs
                           )
    elif model_type == MODEL_TYPE_VAE_SEMI_SUPERVISED_CIFAR10:
        model = SemiSupervisedClassifier(exp_config=exp_config,
                                         sess=sess,
                                         epoch=num_epochs,
                                         dao=dao,
                                         test_data_iterator=test_data_iterator,
                                         check_point_epochs=check_point_epochs
                                         )
    elif model_type == VAE_FCNN:
        model = SemiSupervisedSegmenterMnist(exp_config=exp_config,
                                         sess=sess,
                                         epoch=num_epochs,
                                         dao=dao,
                                         test_data_iterator=test_data_iterator,
                                         check_point_epochs=check_point_epochs
                                         )
    elif model_type == VAE._model_name_:
        model = VAE(exp_config=exp_config,
                    sess=sess,
                    epoch=num_epochs,
                    dao=dao,
                    test_data_iterator=test_data_iterator,
                    check_point_epochs=check_point_epochs
                    )
    elif model_type == MODEL_TYPE_VAE_SEMI_SUPERVISED_MNIST:
        model = SemiSupervisedClassifierMnist(exp_config=exp_config,
                                              sess=sess,
                                              epoch=num_epochs,
                                              dao=dao,
                                              test_data_iterator=test_data_iterator,
                                              check_point_epochs=check_point_epochs,
                                              num_individual_samples_annotated=num_individual_samples_annotated,
                                              num_samples_wrongly_annotated=num_samples_wrongly_annotated,
                                              total_confidence_of_wrong_annotation=total_confidence_of_wrong_annotation
                                              )
    else:
        raise Exception(
            f"Unrecognized model type {model_type}"
            f"Model type Should be one of model_type should be one of {model_types}")
    return model


def train_and_get_features(exp: Experiment,
                           model: Model,
                           train_val_data_iterator: TrainValDataIterator):
    exp.model = model
    # show network architecture
    show_all_variables()

    exp.train(train_val_data_iterator)

    train_val_data_iterator.reset_counter("train")
    train_val_data_iterator.reset_counter("val")
    exp.encode_latent_vector(train_val_data_iterator, "train", save_images=False)

    train_val_data_iterator.reset_counter("train")
    train_val_data_iterator.reset_counter("val")
    return exp.encode_latent_vector(train_val_data_iterator, "val")


def test(exp: Experiment,
         model: VAAL_ARCHITECTURE_FOR_CIFAR,
         data_iterator: DataIterator
         ):
    exp.model = model
    show_all_variables()
    predicted_df = exp.test(data_iterator)
    return predicted_df


def load_model_and_test(experiment_name,
                        z_dim,
                        run_id,
                        model_type,
                        num_cluster_config=None,
                        num_units=None,
                        save_reconstructed_images=True,
                        split_name="Split_1",
                        data_iterator=None,
                        num_val_samples=128,
                        root_path="/Users/sunilv/concept_learning_exp",
                        dataset_name="mnist",
                        write_predictions=True
                        ):
    if num_units is None:
        num_units = [64, 128, 32]
    dao = get_dao(dataset_name, split_name, num_val_samples)
    exp_config = ExperimentConfig(root_path=root_path,
                                  num_decoder_layer=len(num_units) + 1,
                                  z_dim=z_dim,
                                  num_units=num_units,
                                  num_cluster_config=num_cluster_config,
                                  confidence_decay_factor=5,
                                  dataset_name=dataset_name,
                                  split_name=split_name,
                                  model_name="VAE",
                                  batch_size=64,
                                  name=experiment_name,
                                  num_val_samples=num_val_samples,
                                  save_reconstructed_images=save_reconstructed_images,
                                  write_predictions=write_predictions
                                  )
    exp_config.check_and_create_directories(run_id, create=False)
    exp = Experiment(1, experiment_name, exp_config, run_id)
    print(exp.as_json())
    if data_iterator is None:
        data_iterator = DataIterator(dataset_path=exp.config.DATASET_PATH,
                                     batch_size=exp.config.BATCH_SIZE,
                                     dao=dao)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        model = get_model(dao, exp_config, model_type, num_epochs=-1, sess=sess, test_data_iterator=data_iterator)

        print("Starting Inference")
        predicted_df = test(exp, model, data_iterator)
        data_iterator.reset_counter("test")
        return exp_config, predicted_df


if __name__ == "__main__":
    model_1, exp_config_1, _, _, epochs_completed = load_trained_model(experiment_name="Experiment_4",
                                                                       z_dim=32,
                                                                       run_id=3,
                                                                       num_cluster_config=ExperimentConfig.NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW,
                                                                       manual_labels_config=ExperimentConfig.USE_ACTUAL,
                                                                       supervise_weight=0,
                                                                       beta=0,
                                                                       reconstruction_weight=1,
                                                                       model_type=MODEL_TYPE_VAE_UNSUPERVISED_CIFAR10,
                                                                       num_units=[64, 128, 64, 64],
                                                                       save_reconstructed_images=True,
                                                                       split_name="split_1",
                                                                       num_val_samples=128,
                                                                       learning_rate=0.001,
                                                                       dataset_name="cifar_10",
                                                                       activation_output_layer="LINEAR",
                                                                       write_predictions=False,
                                                                       seed=547,
                                                                       root_path="/Users/sunilv/concept_learning_exp",
                                                                       eval_interval_in_epochs=1
                                                                       )
    print("Number of epochs completed", model_1.num_training_epochs_completed)
