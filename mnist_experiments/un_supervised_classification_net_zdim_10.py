import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features
experiment_name = "un_supervised_classification_z_dim_10"
num_epochs = 10
create_split = False
z_dim = 10
for run_id in range(2, 5):
    initialize_model_train_and_get_features(experiment_name, z_dim, run_id, create_split, num_epochs)
    tf.reset_default_graph()
