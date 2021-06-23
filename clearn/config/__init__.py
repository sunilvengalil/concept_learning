import os
import json
import logging

from clearn.dao.dao_factory import get_dao
from clearn.utils.dir_utils import check_and_create_folder
ROOT_PATH = "/Users/sunilkumar/concept_learning_exp/"

N_3_DEFAULT = 32
N_2_DEFAULT = 64
N_1_DEFAULT = 20
Z_DIM_DEFAULT = 20

WEIGHTS = "Weights"
BIAS = "Bias"
LAYER_NAME_PREFIX = "Layer"


def get_keys(base_path, key_prefix):
    keys = []
    for file in os.listdir(base_path):
        if os.path.isdir(base_path + file):
            if file.startswith(key_prefix):
                keys.append(file)
    return keys


def get_base_path(exp_config,
                  run_id: int = 0
                  ) -> str:
    """
    :rtype:
    """

    num_units = exp_config.num_units
    if len(exp_config.num_units) >= 3:
        units_ = str(exp_config.num_units[-1])
        for i in exp_config.num_units[1:-1][::-1]:
            units_ += "_" + str(i)
    else:
        if len(num_units) == 2:
            units_ = "0"
        else:
            units_ = "0_0"
    if exp_config.num_cluster_config is None:
        return os.path.join(os.path.join(exp_config.root_path, exp_config.name),
                            f"Exp_{units_}_{num_units[0]}_{exp_config.Z_DIM}_{run_id}/")
    else:
        return os.path.join(os.path.join(exp_config.root_path, exp_config.name),
                            f"Exp_{units_}_{num_units[0]}_{exp_config.Z_DIM}_{exp_config.num_cluster_config}_{run_id}/")


class ExperimentConfig:
    NUM_CLUSTERS_CONFIG_ELBOW = "ELBOW"
    NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW = "TWO_TIMES_ELBOW"
    USE_ACTUAL = "USE_ACTUAL"
    USE_CLUSTER_CENTER = "USE_CLUSTER_CENTER"
    DISTANCE_EUCLIDEAN = "EUCLIDEAN"
    DISTANCE_MAHALANOBIS = "MAHALANOBIS"
    CLUSTERING_K_MEANS = "K_MEANS"
    CLUSTERING_GMM = "GMM"
    CONFIDENCE_DECAY_FUNCTION_EXPONENTIAL = "EXPONENTIAL"
    CONFIDENCE_DECAY_FUNCTION_GAUSSIAN = "GAUSSIAN"

    def __init__(self,
                 root_path,
                 num_decoder_layer,
                 z_dim,
                 num_units,
                 num_cluster_config,
                 strides,
                 num_dense_layers,
                 confidence_decay_factor=2,
                 beta=5,
                 supervise_weight=0,
                 dataset_name="mnist",
                 split_name="Split_1",
                 model_name="VAE",
                 batch_size=64,
                 name="experiment_configuration",
                 num_val_samples=128,
                 total_training_samples=60000,
                 manual_labels_config=USE_CLUSTER_CENTER,
                 reconstruction_weight=1,
                 activation_hidden_layer="RELU",
                 activation_output_layer="SIGMOID",
                 save_reconstructed_images=True,
                 learning_rate=0.005,
                 max_checkpoints_to_keep=20,
                 beta1_adam=0.9,
                 run_evaluation_during_training=True,
                 model_save_interval=1,
                 write_predictions=True,
                 eval_interval_in_epochs=1,
                 return_latent_vector=True,
                 budget=1,
                 seed=547,
                 distance_metric=DISTANCE_EUCLIDEAN,
                 clustering_alg=CLUSTERING_K_MEANS,
                 confidence_decay_function=CONFIDENCE_DECAY_FUNCTION_EXPONENTIAL,
                 log_level=logging.INFO,
                 fully_convolutional=False,
                 num_concepts=10,
                 supervise_weight_concepts=1,
                 uncorrelated_features=False,
                 env=None,
                 translate_image=False,
                 dao=None,
                 concept_id=-1,
                 concept_dict=None
                 ):
        """
        :param manual_labels_config: str Specifies whether to use actual label vs cluster center label
        :rtype: object
        :type num_cluster_config: str
        :type num_units: list
        :type beta: float
        """
        # if ExperimentConfig._instance is not None:
        #     raise Exception("ExperimentConfig is singleton class. Use class method get_exp_config() instead")
        if root_path is not None :
            # TODO validate if the folder can be created or not
            self.root_path = root_path
        elif env is not None:
            if env == "sunil_local":
                self.root_path = "/Users/sunilv/concept_learning_exp"
            elif env == "colab":
                self.root_path = "/content/gdrive/MyDrive/concept_learning/concept_learning/concept_learning_exp"
            else:
                raise Exception(f"Parameter env should be set as sunil_local or colab. env is passed as {env} instead")

        if len(num_units) < 1 :
            print(num_units)
            raise ValueError("Length of num_units should be greater than 1")

        self.learning_rate = learning_rate
        self.num_decoder_layer = len(num_units) + 1
        self.Z_DIM = z_dim
        self.num_units = num_units
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.model_name = model_name
        self.BATCH_SIZE = batch_size
        self.beta = beta
        self.supervise_weight = supervise_weight
        self.MODEL_NAME_WITH_CONFIG = "{}_{}_{:2d}_{:02d}".format(model_name,
                                                                  dataset_name,
                                                                  self.BATCH_SIZE,
                                                                  z_dim)
        _dataset_name = dataset_name
        if _dataset_name =="mnist_concepts":
            _dataset_name = "mnist"
        self.DATASET_ROOT_PATH = os.path.join(self.root_path, "datasets/" + _dataset_name)
        self.name = name
        self.num_val_samples = num_val_samples
        self.num_cluster_config = num_cluster_config
        self.confidence_decay_factor = confidence_decay_factor
        self.manual_labels_config = manual_labels_config
        self.reconstruction_weight = reconstruction_weight
        if dao is None:
            # base_path = get_base_path()
            #
            self.dao = get_dao(dataset_name,
                          split_name,
                               num_val_samples,
                               dataset_path=os.path.join(self.root_path, "datasets/"),
                               concept_id = concept_id
                          )
        else:
            self.dao = dao
        # self.num_train_samples = (self.dao.number_of_training_samples  // batch_size) * batch_size
        self.activation_hidden_layer = activation_hidden_layer
        self.activation_output_layer = activation_output_layer
        self.save_reconstructed_images = save_reconstructed_images
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.beta1_adam = beta1_adam
        self.run_evaluation_during_training = run_evaluation_during_training
        self.model_save_interval = model_save_interval
        self.write_predictions = write_predictions
        self.eval_interval_in_epochs = eval_interval_in_epochs
        self.return_latent_vector = return_latent_vector
        self.seed = seed
        self.budget = budget
        self.clustering_alg = clustering_alg
        self.confidence_decay_function = confidence_decay_function
        self.distance_metric = distance_metric
        self.log_level = log_level
        self.fully_convolutional = fully_convolutional
        self.num_concepts = num_concepts
        self.supervise_weight_concepts = supervise_weight_concepts
        self.strides = strides
        self.num_dense_layers = num_dense_layers
        self.uncorrelated_features = uncorrelated_features
        self.translate_image = translate_image
        self.concept_id = concept_id
        self.concept_dict = concept_dict

    @property
    def num_train_samples(self):
        return (self.dao.number_of_training_samples // self.BATCH_SIZE) * self.BATCH_SIZE

    def set_root_path(self, env):
        if env == "sunil_local":
            self.root_path = "/Users/sunilv/concept_learning_exp"
        elif env == "colab":
            self.root_path = "/content/gdrive/MyDrive/concept_learning/concept_learning/concept_learning_exp"

    def as_json(self):
        config_json = dict()
        config_json["NUM_UNITS"] = self.num_units
        config_json["ROOT_PATH"] = self.root_path
        config_json["NUM_DECODER_LAYER"] = self.num_decoder_layer
        config_json["Z_DIM"] = self.Z_DIM
        config_json["BETA"] = self.beta
        config_json["DATASET_NAME"] = self.dataset_name
        config_json["MODEL_NAME"] = self.model_name
        config_json["BATCH_SIZE"] = self.BATCH_SIZE
        config_json["SPLIT_NAME"] = self.split_name
        config_json["SUPERVISE_WEIGHT"] = self.supervise_weight
        config_json["BETA"] = self.beta
        config_json["NUM_CLUSTER_CONFIG"] = self.num_cluster_config
        config_json["CONFIDENCE_DECAY_FACTOR"] = self.confidence_decay_factor
        config_json["MANUAL_LABELS_CONFIG"] = self.manual_labels_config
        config_json["RECONSTRUCTION_WEIGHT"] = self.reconstruction_weight
        config_json["ACTIVATION_HIDDEN_LAYER"] = self.activation_hidden_layer
        config_json["ACTIVATION_OUTPUT_LAYER"] = self.activation_output_layer
        config_json["SAVE_RECONSTRUCTED_IMAGES"] = self.save_reconstructed_images
        config_json["NUM_VAL_SAMPLES"] = self.num_val_samples
        config_json["LEARNING_RATE"] = self.learning_rate
        config_json["MAX_CHECKPOINTS_TO_KEEP"] = self.max_checkpoints_to_keep
        config_json["BETA1_ADAM"] = self.beta1_adam
        config_json["RUN_EVALUATION_DURING_TRAINING"] = self.run_evaluation_during_training
        config_json["MODEL_SAVE_INTERVAL"] = self.model_save_interval
        config_json["WRITE_PREDICTIONS"] = self.write_predictions
        config_json["EVAL_INTERVAL_IN_EPOCHS"] = self.eval_interval_in_epochs
        config_json["RETURN_LATENT_VECTOR"] = self.return_latent_vector
        config_json["SEED"] = self.seed
        config_json["BUDGET"] = self.budget
        config_json["CLUSTERING_ALG"] = self.clustering_alg
        config_json["CONFIDENCE_DECAY_FUNCTION"] = self.confidence_decay_function
        config_json["DISTANCE_METRIC"] = self.distance_metric
        config_json["LOG_LEVEL"] = self.log_level
        config_json["FULLY_CONVOLUTIONAL"] = self.fully_convolutional
        config_json["NUM_CONCEPTS"] = self.num_concepts
        config_json["SUPERVISE_WEIGHT_CONCEPTS"] = self.supervise_weight_concepts
        config_json["STRIDES"] = self.strides
        config_json["NUM_DENSE_LAYER"] = self.num_dense_layers
        config_json["UNCORRELATED_FEATURES"] = self.uncorrelated_features
        config_json["TRANSLATE_IMAGE"] = self.translate_image
        config_json["CONCEPT_ID"] = self.concept_id
        config_json["CONCEPT_DICT"] = self.concept_dict

        return config_json

    def get_exp_name_with_parameters(self, run_id):
        if self.num_cluster_config is None:
            return f"Exp_{self.num_units[3]}_{self.num_units[2]}_{self.num_units[1]}_{self.num_units[0]}_{self.Z_DIM}_{run_id}/"
        else:
            return f"Exp_{self.num_units[3]}_{self.num_units[2]}_{self.num_units[1]}_{self.num_units[0]}_{self.Z_DIM}_{self.num_cluster_config}_{run_id}/"

    def _check_and_create_directories(self, run_id, create):
        self.BASE_PATH = get_base_path(self,
                                       run_id=run_id
                                       )

        self.TRAINED_MODELS_PATH = os.path.join(self.BASE_PATH, "trained_models/")

        if self.dataset_name == "mnist_concepts":
            dataset_temp = os.path.join(self.root_path, "datasets/" + self.dataset_name)
            self.DATASET_PATH = os.path.join(dataset_temp, self.split_name + "/")
        else:
            self.DATASET_PATH = os.path.join(self.DATASET_ROOT_PATH, self.split_name + "/")

        self.PREDICTION_RESULTS_PATH = os.path.join(self.BASE_PATH, "prediction_results/")
        self.reconstructed_images_path = os.path.join(self.PREDICTION_RESULTS_PATH, "reconstructed_images/")
        self.LOG_PATH = os.path.join(self.BASE_PATH, "logs/")
        self.ANALYSIS_PATH = os.path.join(self.BASE_PATH, "analysis/")
        paths = [self.root_path,
                 self.BASE_PATH,
                 self.DATASET_ROOT_PATH,
                 self.ANALYSIS_PATH,
                 self.DATASET_PATH,
                 self.TRAINED_MODELS_PATH,
                 self.PREDICTION_RESULTS_PATH,
                 self.reconstructed_images_path,
                 self.LOG_PATH]

        if create:
            list(map(check_and_create_folder, paths))
            return True
        else:
            directories_present = list(map(os.path.isdir, paths))
            missing_directories = [path for i, path in enumerate(paths) if not directories_present[i]]
            if len(missing_directories) > 0:
                print("Missing directories")
                print(missing_directories)
                return False
            else:
                return True

    def check_and_create_directories(self, run_ids, create=True):
        if isinstance(run_ids, list):
            for run_id in run_ids:
                if not self._check_and_create_directories(run_id, create):
                    return False
            return True
        else:
            return self._check_and_create_directories(run_ids, create)

    def get_annotation_result_path(self, base_path=None):
        if base_path is not None and len(base_path) > 0:
            return os.path.join(base_path, "assembled_annotation/")
        else:
            return os.path.join(self.BASE_PATH, "assembled_annotation/")

    @classmethod
    def load_from_json(cls, experiment_name, json_file_name, num_units, dataset_name, split_name):
        with open(json_file_name) as json_fp:
            exp_config_dict = json.load(json_fp)
        exp_config = cls()
        exp_config.initialize_from_dictionary(exp_config_dict)

        if len(num_units) != exp_config.num_decoder_layer - 1:
            print(num_units, exp_config.num_decoder_layer)
            raise ValueError("No of units should be same as number of layers minus one")

        # exp_config.num_units =  # set this
        # num_units.append(z_dim * 2)
        exp_config.num_units = num_units + [exp_config.Z_DIM * 2]
        exp_config.name = experiment_name

        dao = get_dao(dataset_name, split_name, exp_config.num_val_samples)
        total_training_samples = dao.number_of_training_samples()
        exp_config.num_train_samples = ((total_training_samples - exp_config.num_val_samples) // exp_config.BATCH_SIZE) * exp_config.BATCH_SIZE

        return exp_config

    def initialize_from_dictionary(self, exp_config_dict):

        self.root_path = exp_config_dict["ROOT_PATH"]

        self.num_decoder_layer = exp_config_dict["NUM_DECODER_LAYER"]
        self.Z_DIM = exp_config_dict["Z_DIM"]
        self.dataset_name = exp_config_dict["DATASET_NAME"]
        self.split_name = exp_config_dict["SPLIT_NAME"]
        self.model_name = exp_config_dict["MODEL_NAME"]
        self.BATCH_SIZE = exp_config_dict["BATCH_SIZE"]
        self.beta1_adam = exp_config_dict["BETA1_ADAM"]
        self.beta = exp_config_dict["BETA"]
        self.supervise_weight = exp_config_dict["SUPERVISE_WEIGHT"]
        self.MODEL_NAME_WITH_CONFIG = "{}_{}_{:2d}_{:02d}".format(self.model_name,
                                                                  self.dataset_name,
                                                                  self.BATCH_SIZE,
                                                                  self.Z_DIM)
        self.DATASET_ROOT_PATH = os.path.join(self.root_path, "datasets/" + self.dataset_name)
        self.num_val_samples = exp_config_dict["NUM_VAL_SAMPLES"]
        self.num_cluster_config = exp_config_dict["NUM_CLUSTER_CONFIG"]
        self.confidence_decay_factor = exp_config_dict["CONFIDENCE_DECAY_FACTOR"]
        self.manual_labels_config = exp_config_dict["MANUAL_LABELS_CONFIG"]
        self.reconstruction_weight = exp_config_dict["RECONSTRUCTION_WEIGHT"]
        self.num_val_samples = exp_config_dict["NUM_VAL_SAMPLES"]
        self.activation_hidden_layer = exp_config_dict["ACTIVATION_HIDDEN_LAYER"]
        self.activation_output_layer = exp_config_dict["ACTIVATION_OUTPUT_LAYER"]
        self.save_reconstructed_images = exp_config_dict["SAVE_RECONSTRUCTED_IMAGES"]
        self.max_checkpoints_to_keep = exp_config_dict["MAX_CHECKPOINTS_TO_KEEP"]
        self.run_evaluation_during_training = exp_config_dict["RUN_EVALUATION_DURING_TRAINING"]
        self.model_save_interval = exp_config_dict["MODEL_SAVE_INTERVAL"]
        self.write_predictions = exp_config_dict["WRITE_PREDICTIONS"]
        self.eval_interval_in_epochs = exp_config_dict["EVAL_INTERVAL_IN_EPOCHS"]
        self.return_latent_vector = exp_config_dict["RETURN_LATENT_VECTOR"]
        self.seed = exp_config_dict["SEED"]
        self.budget = exp_config_dict["BUDGET"]
        self.confidence_decay_function = exp_config_dict["CONFIDENCE_DECAY_FUNCTION"]
        self.distance_metric = exp_config_dict["DISTANCE_METRIC"]
        self.clustering_alg = exp_config_dict["CLUSTERING_ALG"]
        self.log_level = exp_config_dict["LOG_LEVEL"]
        self.fully_convolutional = exp_config_dict["FULLY_CONVOLUTIONAL"]
        self.num_concepts = exp_config_dict["NUM_CONCEPTS"]
        self.supervise_weight_concepts = exp_config_dict["SUPERVISE_WEIGHT_CONCEPTS"]
        self.strides = exp_config_dict["STRIDES"],
        self.num_dense_layers = exp_config_dict["NUM_DENSE_LAYERS"]
        self.num_dense_layers = exp_config_dict["UNCORRELATED_FEATURES"]
        self.translate_image = exp_config["TRANSLATE_IMAGE"]
        self.concept_id = exp_config["CONCEPT_ID"]
        self.concept_dict = exp_config["CONCEPT_DICT"]

if __name__ == "__main__":
    _root_path = "/Users/sunilv/concept_learning_exp"
    exp_config = ExperimentConfig(root_path=_root_path,
                                  num_decoder_layer=5,
                                  z_dim=32,
                                  num_units=[128, 256, 512, 1024],
                                  num_cluster_config=None,
                                  confidence_decay_factor=5,
                                  beta=5,
                                  supervise_weight=1,
                                  dataset_name="cifar_10",
                                  split_name="split_1",
                                  model_name="VAE",
                                  batch_size=64,
                                  eval_interval_in_epochs=1,
                                  name="Experiment_3",
                                  num_val_samples=5000,
                                  )
    print(get_base_path(exp_config, 1))
