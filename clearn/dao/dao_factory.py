from  clearn.dao.mnist import MnistDao
from clearn.dao.cifar_10 import CiFar10Dao


def get_dao(dataset_name, split_name: str, num_validation_samples: int, num_training_samples: int = -1):
    if dataset_name == "mnist":
        return MnistDao(split_name, num_validation_samples)
    elif dataset_name == "cifar_10":
        return CiFar10Dao(split_name, num_validation_samples, num_training_samples)
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")
