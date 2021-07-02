import numpy as np

from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.models.model import Model
from tensorflow.compat.v1 import Session

from clearn.utils.data_loader import TrainValDataIterator


class GenerativeModel(Model):
    _model_name_ = "GenerativeModel"

    def __init__(self, exp_config: ExperimentConfig,
                 sess: Session,
                 epoch: int,
                 dao: IDao,
                 test_data_iterator):
        super().__init__(exp_config, sess, epoch, dao, test_data_iterator)

    def encode(self, images: np.ndarray):
        pass

    def decode(self, z):
        pass

    def train(self, train_val_data_iterator: TrainValDataIterator):
        pass

    def evaluate(self, train_val_data_iterator: TrainValDataIterator, dataset_type: str):
        pass

    def encode_and_get_features(self, images: np.ndarray):
        pass

    def decode_and_get_features(self, z: np.ndarray):
        pass