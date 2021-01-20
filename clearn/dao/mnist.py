from clearn.dao.idao import IDao
class MnistDao(IDao):

    def number_of_training_samples(self):
        return 60000

    def __init__(self):
        self.dataset_name = "mnist"
