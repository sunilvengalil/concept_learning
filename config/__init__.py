import os
from utils.utils import check_folder

WEIGHTS = "Weights"
BIAS =  "Bias"
LAYER_NAME_PREFIX = "Layer"

def get_base_path(root_path, z_dim, n_3, n_2, version = ""):
    return os.path.join(root_path, "Exp_{:02d}_{:03}_{:03d}_{}/".format(z_dim, n_3, n_2, version))

class ExperimentConfig:
    def __init__(self, root_path, num_decoder_layer, z_dim, num_units,
                 beta = 5,
                 supervise_weight = 0,
                 dataset_name="mnist",split_name="Split_1",
                 model_name="VAE",
                 batch_size=64,

                 ):
        if len(num_units) != num_decoder_layer - 1:
            raise ValueError("No of units should be same as number of layers minus one")
        num_units.append(z_dim * 2)
        self.num_decoder_layer = num_decoder_layer
        self.Z_DIM = z_dim
        self.num_units = num_units
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.model_name = model_name
        self.BATCH_SIZE = batch_size
        self.root_path = root_path
        self.beta = beta
        self.MODEL_NAME_WITH_CONFIG = "{}_{}_{:2d}_{:02d}".format(model_name,
                                                                  dataset_name,
                                                                  self.BATCH_SIZE,
                                                                  z_dim)
    def asJson(self):
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

        return config_json



    def create_directories(self,run_id):
        self.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS = os.path.join(self.root_path, "datasets/" + self.dataset_name)

        self.BASE_PATH = get_base_path(self.root_path, self.Z_DIM, self.num_units[self.num_decoder_layer - 2],
                                  self.num_units[self.num_decoder_layer - 3],
                                  version=run_id)
        DATASET_ROOT_PATH = os.path.join(self.BASE_PATH, self.dataset_name + "/")
        self.DATASET_PATH = os.path.join(DATASET_ROOT_PATH, self.split_name + "/")

        self.MODEL_PATH = os.path.join(self.DATASET_PATH, self.MODEL_NAME_WITH_CONFIG)
        self.SPLIT_PATH = os.path.join(self.DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS, self.split_name + "/")
        self.TRAINED_MODELS_PATH = os.path.join(self.MODEL_PATH, "trained_models/")
        self.PREDICTION_RESULTS_PATH = os.path.join(self.MODEL_PATH, "prediction_results/")
        self.LOG_PATH = os.path.join(self.MODEL_PATH,"logs/")
        self.ANALYSIS_PATH = os.path.join(self.MODEL_PATH, "analysis/")
        check_folder(self.ANALYSIS_PATH)

        if not os.path.isdir(self.BASE_PATH):
            print("Creating directory{}".format(self.BASE_PATH))
            os.mkdir(self.BASE_PATH)
        if not os.path.isdir(DATASET_ROOT_PATH):
            print("Creating directory{}".format(DATASET_ROOT_PATH))
            os.mkdir(DATASET_ROOT_PATH)
        if not os.path.isdir(self.DATASET_PATH):
            print("Creating directory{}".format(self.DATASET_PATH))
            os.mkdir(self.DATASET_PATH)
        if not os.path.isdir(self.MODEL_PATH):
            print("Creating directory{}".format(self.MODEL_PATH))
            os.mkdir(self.MODEL_PATH)
        if not os.path.isdir(self.SPLIT_PATH):
            print("Creating directory{}".format(self.SPLIT_PATH))
            os.mkdir(self.SPLIT_PATH)
        if not os.path.isdir(self.TRAINED_MODELS_PATH):
            os.mkdir(self.TRAINED_MODELS_PATH)
        if not os.path.isdir(self.PREDICTION_RESULTS_PATH):
            os.mkdir(self.PREDICTION_RESULTS_PATH)
        if not os.path.isdir(self.LOG_PATH):
            print("Creating directory{}".format(self.LOG_PATH))
            os.mkdir(self.LOG_PATH)
