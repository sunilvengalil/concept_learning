import os

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
                 dao: IDao):
        super().__init__(exp_config, sess, epoch, dao)

    def encode(self, images):
        pass

    def decode(self, z):
        pass

    def train(self, train_val_data_iterator: TrainValDataIterator):
        pass

    def evaluate(self, train_val_data_iterator: TrainValDataIterator, dataset_type: str):
        pass
