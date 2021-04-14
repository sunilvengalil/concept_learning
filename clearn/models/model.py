from abc import ABC
import os
import tensorflow as tf

from tensorflow.compat.v1 import Session

from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.utils.data_loader import TrainValDataIterator, DataIterator


class Model(ABC):
    _model_name_ = "Model"
    dataset_type_test = "test"
    dataset_type_train = "train"
    dataset_type_val = "val"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess: Session,
                 epoch: int,
                 dao: IDao,
                 test_data_iterator: DataIterator = None):
        self.exp_config = exp_config
        self.sess = sess
        self.epoch = epoch
        self.dao = dao
        self.test_data_iterator = test_data_iterator


    def _initialize(self,
                    restore_from_existing_checkpoint=True,
                    check_point_epochs=None):
        # saver to save model
        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)
        # # summary writer
        # self.writer = tf.compat.v1.summary.FileWriter(self.exp_config.LOG_PATH + '/' + self._model_name_,
        #                                     self.sess.graph)
        # self.writer_v = tf.compat.v1.summary.FileWriter(self.exp_config.LOG_PATH + '/' + self._model_name_ + "_v",
        #                                       self.sess.graph)

        if restore_from_existing_checkpoint:
            # restore check-point if it exits
            could_load, checkpoint_counter = self._load(self.exp_config.TRAINED_MODELS_PATH,
                                                        check_point_epochs=check_point_epochs)
            if could_load:
                num_batches_train = self.dao.number_of_training_samples // self.exp_config.BATCH_SIZE
                start_epoch = int(checkpoint_counter / num_batches_train)
                start_batch_id = checkpoint_counter - start_epoch * num_batches_train
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                start_epoch = 0
                start_batch_id = 0
                counter = 1
                print(" [!] Load failed...")
        else:
            counter = 1
            start_epoch = 0
            start_batch_id = 0
        return counter, start_batch_id, start_epoch

    def _load(self, checkpoint_dir, check_point_epochs=None):
        import re
        # saver to save model
        self.saver = tf.compat.v1.train.Saver(max_to_keep=20)

        checkpoint_dir = checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(f"Reading checkpoints from {checkpoint_dir} State {ckpt} ")

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if check_point_epochs is not None:
                num_training_samples = self.dao.number_of_training_samples // self.exp_config.BATCH_SIZE
                print("num_training_samples", num_training_samples)
                if check_point_epochs == 1:
                    steps = check_point_epochs * num_training_samples + 1
                else:
                    steps = check_point_epochs * num_training_samples
                ckpt_name = f"{self._model_name_}.model-{steps}"
            print("ckpt_name", ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train(self, train_val_data_iterator: TrainValDataIterator):
        pass

    def evaluate(self,
                 val_data_iterator: DataIterator,
                 epoch: int,
                 dataset_type: str,
                 return_latent_vector: bool):
        pass

    def load(self):
        pass

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self._model_name_ + '.model'), global_step=step)
