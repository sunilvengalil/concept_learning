import os
from utils.dir_utils import check_and_create_folder

WEIGHTS = "Weights"
BIAS = "Bias"
LAYER_NAME_PREFIX = "Layer"


def get_base_path(root_path: str,
                  z_dim: int,
                  n_3: int,
                  n_2: int,
                  cluster_config,
                  version: int = 0
                  ) -> str:

    return os.path.join(root_path, f"Exp_{z_dim:02d}_{n_3:03}_{n_2:03d}_{cluster_config}_{version}/")


class ExperimentConfig:
    NUM_CLUSTERS_CONFIG_ELBOW = "ELBOW"
    NUM_CLUSTERS_CONFIG_TWO_TIMES_ELBOW = "TWO_TIMES_ELBOW"

    def __init__(self, root_path, num_decoder_layer, z_dim, num_units,
                 num_cluster_config,
                 beta=5,
                 supervise_weight=0,
                 dataset_name="mnist",
                 split_name="Split_1",
                 model_name="VAE",
                 batch_size=64,
                 name="experiment_configuration",
                 num_val_samples=128,
                 total_training_samples=60000,
                 ):
        """
        :type num_cluster_config: str
        :type num_units: list
        :type beta: float
        """
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
        self.supervise_weight = supervise_weight
        self.MODEL_NAME_WITH_CONFIG = "{}_{}_{:2d}_{:02d}".format(model_name,
                                                                  dataset_name,
                                                                  self.BATCH_SIZE,
                                                                  z_dim)
        self.DATASET_ROOT_PATH = os.path.join(self.root_path, "datasets/" + dataset_name)
        self.name = name
        self.num_val_samples = num_val_samples
        self.num_cluster_config = num_cluster_config
        self.num_train_samples = ((total_training_samples - num_val_samples) // batch_size) * batch_size

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
        return config_json

    def _check_and_create_directories(self, run_id, create):
        self.BASE_PATH = get_base_path(self.root_path, self.Z_DIM, self.num_units[self.num_decoder_layer - 2],
                                       self.num_units[self.num_decoder_layer - 3],
                                       self.num_cluster_config,
                                       version=run_id
                                       )
        self.DATASET_PATH = os.path.join(self.DATASET_ROOT_PATH, self.split_name + "/")

        self.TRAINED_MODELS_PATH = os.path.join(self.BASE_PATH, "trained_models/")
        self.PREDICTION_RESULTS_PATH = os.path.join(self.BASE_PATH, "prediction_results/")
        self.LOG_PATH = os.path.join(self.BASE_PATH, "logs/")
        self.ANALYSIS_PATH = os.path.join(self.BASE_PATH, "analysis/")
        paths = [self.root_path, self.BASE_PATH,
                 self.DATASET_ROOT_PATH, self.ANALYSIS_PATH,
                 self.DATASET_PATH,
                 self.TRAINED_MODELS_PATH, self.PREDICTION_RESULTS_PATH,
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
