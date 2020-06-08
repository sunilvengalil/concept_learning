import os
import gzip
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import json


def load_mnist(dataset_name):
    data_dir = os.path.join("/Users/sunilkumar/gitprojects/tensorflow-generative-model-collections/", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    train_x = data.reshape((60000, 28, 28, 1))

    #code to save images
    for i in range(10):
        cv2.imwrite('lol' + str(i) + '.jpg', train_x[i])

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    train_y = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    test_x = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    test_y = data.reshape((10000))

    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)

    X = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y, test_y), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


class TrainValDataIterator:
    @classmethod
    def load_train_val(cls, split_name, split_location):
        with open(split_location + split_name + ".json") as fp:
            dataset_dict = json.load(fp)

        split_names = dataset_dict["split_names"]
        for split in split_names:
            df = pd.read_csv(split_location + split + ".csv")
            dataset_dict[split] = df
        split_names = dataset_dict["split_names"]
        #TODO fix this remove hard coding of split name and column names
        train = dataset_dict["train"]
        x_columns = list(train.columns)
        x_columns.remove('label')
        data = train[x_columns].values
        train_x = data.reshape((data.shape[0], 28, 28, 1))
        data = train[['label']].values
        train_y = np.asarray(data.reshape(data.shape[0])).astype(np.int)

        val = dataset_dict["validation"]
        val_x = val[x_columns].values
        val_x = val_x.reshape((val_x.shape[0], 28, 28, 1))

        val_y = val[['label']].values
        val_y = np.asarray(val_y.reshape(val_y.shape[0])).astype(np.int)


        if len(split_names) != 2:
            raise Exception("Split not implemented for for than two splits")

        _val_y = np.zeros((len(val_y), 10), dtype=np.float)
        for i, label in enumerate(val_y):
            _val_y[i, val_y[i]] = 1.0

        _train_y = np.zeros((len(train_y), 10), dtype=np.float)
        for i, label in enumerate(train_y):
            _train_y[i, train_y[i]] = 1.0

        # return train_x / 255., _train_y, val_x / 255., _val_y
        #TODO separate normalizing and loading logic
        return {"train_x": train_x / 255., "train_y": _train_y, "validation_x": val_x / 255., "validation_y": _val_y}
        #return {"train_x": train_x, "train_y": _train_y, "validation_x": val_x, "validation_y": _val_y}

    @classmethod
    def from_existing_split(cls, split_name, split_location,batch_size =  None):
        instance = cls()
        instance.batch_size = batch_size
        instance.dataset_dict = cls.load_train_val(split_name, split_location)

        instance.train_x = instance.dataset_dict["train" + "_x"]
        instance.train_y = instance.dataset_dict["train" + "_y"]
        instance.val_x = instance.dataset_dict["validation" + "_x"]
        instance.val_y = instance.dataset_dict["validation" + "_y"]

        instance.train_idx = 0
        instance.val_idx = 0

        return instance


    def __init__(self, dataset_path=None, shuffle = False,
                 stratified = None,
                 validation_samples = 100,
                 split_location=None,
                 split_names=[],
                 batch_size=None
                 ):
        self.train_idx = 0
        self.val_idx = 0
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        if dataset_path is not None :
            self.entire_data_x, self.entire_data_y = load_train(dataset_path)
            percentage_to_be_sampled = validation_samples / len(self.entire_data_y)

            # self.train_x, self.train_y, self.val_x, self.val_y = load_train_val(dataset_path,
            #                                                                     shuffle = shuffle,
            #                                                                     stratified = stratified,
            #                                                                     percentage_to_be_sampled = percentage_to_be_sampled,
            #                                                                     split_location=split_location,
            #                                                                     split_names=split_names)
            self.dataset_dict = load_train_val(dataset_path, shuffle = shuffle,
                                                stratified = stratified,
                                                percentage_to_be_sampled = percentage_to_be_sampled,
                                                split_location=split_location,
                                                split_names=split_names)
            #TODO remove this later start using dataset_dict instead
            self.train_x = self.dataset_dict["train" + "_x"]
            self.train_y = self.dataset_dict["train" + "_y"]
            self.val_x = self.dataset_dict["validation" + "_x"]
            self.val_y = self.dataset_dict["validation" + "_y"]


    # def get_next_batch(self, batch_size, split_name):
    #     x = train_x[self.train_idx * batch_size:(self.train_idx + 1) * batch_size]
    #     self.train_idx += 1
    #     return x

    def has_next_val(self):
        #TODO fix this to handle last batch
        return self.val_idx * self.batch_size < self.val_x.shape[0]


    def has_next_train(self):
        #TODO fix this to handle last batch
        return self.train_idx * self.batch_size - self.train_x.shape[0] < self.batch_size


    def get_feature_shape(self):
        return self.val_x.shape[1:]


    def get_next_batch_train(self):
        if self.batch_size is None:
            raise Exception("batch_size attribute is not set")
        x = self.train_x[self.train_idx * self.batch_size:(self.train_idx + 1) * self.batch_size]
        self.train_idx += 1
        return x

    def get_num_samples_train(self):
        return len(self.train_x)

    def get_next_batch_val(self):
        if self.batch_size is None:
            raise Exception("batch_size attribute is not set")
        x = self.val_x[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
        y = self.val_y[self.val_idx * self.batch_size:(self.val_idx + 1) * self.batch_size]
        # TODO check if this is last batch, if yes,reset the counter

        self.val_idx += 1


        return x,y

    def get_num_samples_val(self):
        return len(self.val_x)

    def reset_train_couner(self):
        self.train_idx = 0

    def reset_val_couner(self):
        self.val_idx = 0


def load_train(data_dir, directory_for_sample_images = None,
               shuffle = True):
    data_dir = os.path.join(data_dir , "images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    #code to save images
    if directory_for_sample_images is not None:
        for i in range(10):
            cv2.imwrite(directory_for_sample_images+ str(i) + '.jpg', trX[i])

    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    trY = np.asarray(trY).astype(np.int)

    if shuffle:
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(trX)
        np.random.seed(seed)
        np.random.shuffle(trY)

    y_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec[i, trY[i]] = 1.0

    return trX / 255., y_vec

def load_test(data_dir, shuffle = False, stratified = True, percentage_to_be_sampled = 1):
    teX, teY = load_test_raw_data(data_dir)

    teY = np.asarray(teY).astype(np.int)

    if shuffle:
        seed = 547
        np.random.seed(seed)
        if stratified:
            teX,_,teY,_ = train_test_split(teX,teY,train_size=percentage_to_be_sampled,
                                           stratify= teY)

        else:
            #TODO incorporate parameter % to be samples
            np.random.shuffle(teX)
            np.random.seed(seed)
            np.random.shuffle(teY)

    y_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        y_vec[i, teY[i]] = 1.0

    return teX / 255., y_vec


def load_test_raw_data(data_dir):
    data_dir = os.path.join(data_dir, "images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))
    return teX, teY


def load_train_val(data_dir, shuffle = False,
                   stratified = None,
                   percentage_to_be_sampled = 0.7,
                   split_location=None,
                   split_names = []):

    data_dir = os.path.join(data_dir ,"images/")

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    x = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    y = np.asarray(data.reshape(60000)).astype(np.int)

    seed = 547
    _stratify = None
    if stratified:
        _stratify = y

    if len(split_names) == 2 :
        splitted = train_test_split(x, y,test_size=percentage_to_be_sampled,
                                    stratify= _stratify, shuffle=shuffle,
                                    random_state=seed)
        train_x = splitted[0]
        val_x = splitted[1]
        train_y = splitted[2]
        val_y = splitted[3]

        # TODO change this to save only indices.
        # Alternately save the seed after verifying that same seed generates same split
        if split_location[-1] == "/":
            split_name = split_location[:-1].rsplit("/", 1)[1]
        else:
            split_name = split_location.rsplit("/", 1)[1]
        dataset_dict = {}
        num_splits = len(split_names)
        dataset_dict["split_names"] = split_names

        for split_num, split in enumerate(split_names):
            # TODO remove hard coding of dimensions below
            train_df = pd.DataFrame(splitted[split_num].reshape(splitted[split_num].shape[0], 28 * 28))
            train_df["label"] = splitted[split_num + num_splits]
            train_df.to_csv(split_location + split + ".csv", index=False)
            #dataset_dict[split]=train_df
        print(split_location)
        json_ = split_location + split_name + ".json"
        with open(json_, "w") as fp:
            print("Writing json to ",json_)
            json.dump(dataset_dict, fp)
        print("Writing json to ", json_)

    else:
        raise Exception("Split not implemented for for than two splits")


    _val_y = np.zeros((len(val_y), 10), dtype=np.float)
    for i, label in enumerate(val_y):
        _val_y[i, val_y[i]] = 1.0

    _train_y = np.zeros((len(train_y), 10), dtype=np.float)
    for i, label in enumerate(train_y):
        _train_y[i, train_y[i]] = 1.0

    #return train_x / 255., _train_y, val_x / 255., _val_y
    #todo separate normalizing and loading logic
    return {"train_x":train_x / 255.,"train_y": _train_y, "validation_x":val_x / 255.,"validation_y": _val_y}
