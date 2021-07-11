from clearn.dao.cat_vs_dog import CatVsDogDao
from  clearn.dao.mnist import MnistDao
from clearn.dao.cifar_10 import CiFar10Dao
from clearn.dao.mnist_with_concepts import MnistConceptsDao


class IDao:
    pass


class DriveDao(IDao):
    pass


def get_dao(dataset_name,
            split_name: str,
            num_validation_samples: int,
            num_training_samples: int = -1,
            analysis_path=None,
            dataset_path=None,
            concept_id=None,
            translate_image=False,
            std_dev=1):
    if dataset_name == "mnist":
        return MnistDao(split_name, num_validation_samples)
    elif dataset_name == "cifar_10":
        return CiFar10Dao(split_name, num_validation_samples, num_training_samples)
    elif dataset_name == "mnist_concepts":
        return MnistConceptsDao(dataset_name,
                                split_name,
                                num_validation_samples,
                                dataset_path,
                                concept_id,
                                translate_image=translate_image,
                                std_dev=std_dev
                                )
    elif dataset_name == "cat_vs_dog":
        return CatVsDogDao(dataset_name,
                           split_name,
                           num_validation_samples,
                           dataset_path,
                           concept_id
                           )
    elif dataset_name == "drive":
        return DriveDao(dataset_name,
                        split_name,
                        num_validation_samples = -1,
                                )
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")
