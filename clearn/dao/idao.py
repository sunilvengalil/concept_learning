from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class IDao(ABC):
    VALIDATION_Y_RAW = "validation_y_raw"
    VALIDATION_Y_ONE_HOT = "validation_y"
    VALIDATION_X = "validation_x"
    TRAIN_Y = "train_y"
    TRAIN_X = "train_x"

    Y_RAW = "test_y"
    Y_ONE_HOT = "test_y_one_hot"
    X = "test_x"

    @property
    @abstractmethod
    def number_of_training_samples(self):
        pass

    @property
    @abstractmethod
    def image_shape(self):
        pass

    @property
    @abstractmethod
    def max_value(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    def load_train(self, data_dir, shuffle):
        pass

    @abstractmethod
    def load_train_val_1(self, data_dir):
        pass

    @abstractmethod
    def load_test_1(self, data_dir):
        pass

    def load_test(self, data_dir,
                  split_location=None,
                  split_names=[] ):
        x, y = self.load_test_1(data_dir)
        dataset_dict = {}
        dataset_dict["split_names"] = split_names

        # for split_num, split in enumerate(split_names):
        #     print(split, x.shape)
        #     feature_dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        #     df = pd.DataFrame(x.reshape(x.shape[0],
        #                                       feature_dim)
        #                             )
        #     df["label"] = y
        #     df.to_csv(split_location + split + ".csv", index=False)
        # print(split_location)
        # split_name = self.get_split_name(split_location)
        # json_ = split_location + split_name + ".json"
        # with open(json_, "w") as fp:
        #     print("Writing json to ", json_)
        #     json.dump(dataset_dict, fp)

        y_one_hot = np.eye(self.num_classes)[y]
        return {"test_x": x / self.max_value,
                "test_y": y,
                "test_y_one_hot": y_one_hot}

    def load_train_val(self, data_dir, shuffle=False,
                       stratified=None,
                       percentage_to_be_sampled=0.7,
                       split_location=None,
                       split_names=[]):
        x, y = self.load_train_val_1(data_dir)
        seed = 547
        _stratify = None
        if stratified:
            _stratify = y

        if len(split_names) == 2:
            splitted = train_test_split(x, y, test_size=percentage_to_be_sampled,
                                        stratify=_stratify, shuffle=shuffle,
                                        random_state=seed)
            train_x = splitted[0]
            val_x = splitted[1]
            train_y = splitted[2]
            val_y = splitted[3]
            print(x.shape)
            print(train_x.shape)

            # TODO change this to save only indices.
            # Alternately save the seed after verifying that same seed generates same split
            split_name = self.get_split_name(split_location)
            dataset_dict = {}
            num_splits = len(split_names)
            dataset_dict["split_names"] = split_names

            # for split_num, split in enumerate(split_names):
            #     print(split, splitted[split_num].shape)
            #     feature_dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
            #     train_df = pd.DataFrame(splitted[split_num].reshape(splitted[split_num].shape[0],
            #                                                         feature_dim)
            #                             )
            #     train_df["label"] = splitted[split_num + num_splits]
            #     train_df.to_csv(split_location + split + ".csv", index=False)
            # print(split_location)
            # json_ = split_location + split_name + ".json"
            # with open(json_, "w") as fp:
            #     print("Writing json to ", json_)
            #     json.dump(dataset_dict, fp)
            # print("Writing json to ", json_)
        else:
            raise Exception("Split not implemented for for than two splits")

        _val_y = np.eye(self.num_classes)[val_y]
        _train_y = np.eye(self.num_classes)[train_y]
        # TODO separate normalizing and loading logic
        return {self.TRAIN_X: train_x / self.max_value,
                self.TRAIN_Y: _train_y,
                self.VALIDATION_X: val_x / self.max_value,
                self.VALIDATION_Y_ONE_HOT: _val_y,
                self.VALIDATION_Y_RAW: val_y}

    def get_split_name(self, split_location):
        if split_location[-1] == "/":
            split_name = split_location[:-1].rsplit("/", 1)[1]
        else:
            split_name = split_location.rsplit("/", 1)[1]
        return split_name

    def load_train_val_existing_split(self, split_name, split_location):
        with open(split_location + split_name + ".json") as fp:
            dataset_dict = json.load(fp)

        split_names = dataset_dict["split_names"]
        for split in split_names:
            df = pd.read_csv(split_location + split + ".csv")
            dataset_dict[split] = df
        split_names = dataset_dict["split_names"]

        # TODO fix this remove hard coding of split name and column names
        train = dataset_dict["train"]
        x_columns = list(train.columns)
        x_columns.remove('label')
        data = train[x_columns].values
        train_x = data.reshape((data.shape[0],
                                self.image_shape[0],
                                self.image_shape[1],
                                self.image_shape[2]))
        data = train[['label']].values
        train_y = np.asarray(data.reshape(data.shape[0])).astype(np.int)

        val = dataset_dict["validation"]
        val_x = val[x_columns].values
        val_x = val_x.reshape(val_x.shape[0],
                              self.image_shape[0],
                              self.image_shape[1],
                              self.image_shape[2]
                              )

        val_y = val[['label']].values
        val_y = np.asarray(val_y.reshape(val_y.shape[0])).astype(np.int)

        if len(split_names) != 2:
            raise Exception("Split not implemented for for than two splits")

        _val_y = np.eye(self.num_classes)[val_y]
        _train_y = np.eye(self.num_classes)[train_y]

        # TODO separate normalizing and loading logic
        return {self.TRAIN_X: train_x / self.max_value,
                self.TRAIN_Y: _train_y,
                self.VALIDATION_X: val_x / self.max_value,
                self.VALIDATION_Y_ONE_HOT: _val_y,
                self.VALIDATION_Y_RAW: val_y}


    def load_from_existing_split(self, split_name, split_location):
        with open(split_location + split_name + ".json") as fp:
            dataset_dict = json.load(fp)

        split_names = dataset_dict["split_names"]
        for split in split_names:
            df = pd.read_csv(split_location + split + ".csv")
            dataset_dict[split] = df
        split_names = dataset_dict["split_names"]
        result_dict = dict()
        for split in split_names:
            data_df = dataset_dict[split]
            columns = list(data_df.columns)
            columns.remove('label')
            data = data_df[columns].values

            x = data.reshape((data.shape[0],
                                    self.image_shape[0],
                                    self.image_shape[1],
                                    self.image_shape[2]))
            data = data_df[['label']].values
            y = np.asarray(data.reshape(data.shape[0])).astype(np.int)
            y_one_hot = np.eye(self.num_classes)[y]

            # TODO separate normalizing and loading logic
            result_dict[split + "_x"] = x / self.max_value
            result_dict[split + "_y"] = y
            result_dict[split + "_y_one_hot"] = y_one_hot

        return result_dict
