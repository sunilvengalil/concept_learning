from abc import ABC
import os
import tensorflow as tf
from tensorflow.compat.v1 import Session

from clearn.config import ExperimentConfig
from clearn.utils.data_loader import TrainValDataIterator


class Model(ABC):
    _model_name_ = "Model"

    def __init__(self,
                 exp_config: ExperimentConfig,
                 sess: Session,
                 epoch: int):
        self.exp_config = exp_config
        self.sess = sess
        self.epoch = epoch

    def _initialize(self, train_val_data_iterator=None,
                    restore_from_existing_checkpoint=True,
                    check_point_epochs=None):
        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=50)
        # summary writer
        self.writer = tf.summary.FileWriter(self.exp_config.LOG_PATH + '/' + self._model_name_,
                                            self.sess.graph)
        self.writer_v = tf.summary.FileWriter(self.exp_config.LOG_PATH + '/' + self._model_name_ + "_v",
                                              self.sess.graph)

        if restore_from_existing_checkpoint:
            # restore check-point if it exits
            could_load, checkpoint_counter = self._load(self.exp_config.TRAINED_MODELS_PATH,
                                                        check_point_epochs=check_point_epochs)
            if could_load:
                if train_val_data_iterator is not None:
                    num_batches_train = train_val_data_iterator.get_num_samples("train") // self.exp_config.BATCH_SIZE
                    start_epoch = int(checkpoint_counter / num_batches_train)
                    start_batch_id = checkpoint_counter - start_epoch * num_batches_train
                else:
                    start_epoch = -1
                    start_batch_id = -1
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

        print(" [*] Reading checkpoints...")
        checkpoint_dir = checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if check_point_epochs is not None:
                ckpt_name = check_point_epochs
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

    def evaluate(self, train_val_data_iterator: TrainValDataIterator, dataset_type: str):
        pass

    def load(self):
        pass

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self._model_name_ + '.model'), global_step=step)
