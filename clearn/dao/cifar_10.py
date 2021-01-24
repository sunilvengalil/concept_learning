import numpy as np
from clearn.dao.idao import IDao
class CiFar10Dao(IDao):
    def __init__(self,split_name):
        self.dataset_name = "cifar_10"
        self.split_name = split_name

    @property
    def number_of_training_samples(self):
        return 50000

    @property
    def image_shape(self):
        return(32, 32, 3)

    @property
    def max_value(self):
        return 255.

    @property
    def num_classes(self):
        return 10

    def load_train(self,data_dir, shuffle):
        tr_x, tr_y = self.load_train_val_1(data_dir)

        if shuffle:
            seed = 547
            np.random.seed(seed)
            np.random.shuffle(tr_y)
            np.random.seed(seed)
            np.random.shuffle(tr_y)

        y_vec = np.eye(self.num_classes)[tr_y]

        return tr_x / self.max_value, y_vec

    def load_train_val_1(self, data_dir):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        data = None
        for batch_no in range(1, 6):
            batch_name = "data_batch_" +str(batch_no)
            data_dict = unpickle(data_dir +"/cifar-10-batches-py" + "/" +batch_name)
            if data is None:
                data = data_dict[b"data"]
                label = data_dict[b"labels"]
            else:
                data = np.concatenate((data, data_dict[b"data"]), axis=0)
                label = np.concatenate((label, data_dict[b"labels"]), axis=0)

        tr_x = data.reshape(
            (self.number_of_training_samples, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        tr_y = label.reshape((self.number_of_training_samples))
        tr_y = np.asarray(tr_y).astype(np.int)
        return tr_x, tr_y
