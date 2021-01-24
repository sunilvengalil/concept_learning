from  clearn.dao.mnist import MnistDao
from clearn.dao.cifar_10 import CiFar10Dao


def get_dao(dataset_name, split_name):
    if dataset_name == "mnist":
        return MnistDao(split_name)
    elif dataset_name == "cifar_10":
        return CiFar10Dao(split_name)
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")
