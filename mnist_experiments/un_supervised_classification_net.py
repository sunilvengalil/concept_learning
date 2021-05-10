import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features

experiment_name = "Experiment_1_test"
#root_path = "/Users/sunilv/concept_learning_exp/"
root_path = "C:/concept_learning_exp/"

z_dim_range = [10, 40, 3]
learning_rate = 0.001
num_epochs = 10
num_runs = 5
create_split = True
completed_z_dims = 0
z_dim = 20
run_id =1

#for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
 #   for run_id in range(num_runs):
num_units_list_5layer = [[512, 256, 256, 128, 128],
                         [512, 256, 256, 128, 64],
                         [512, 256, 128, 128, 64],
                         [512, 256, 128, 64, 64],
                         [512, 256, 128, 128, 64],
                         [512, 256, 128, 128, 64]
                      ]

# Test  cases
## Passed ###
num_units = [512, 256, 128, 64, 32]
strides = [2, 2, 2, 1, 1, 1]
num_dense_layers = 2

# num_units = [512, 256, 128, 64]
# strides = [2, 2, 2, 1, 1]
# num_dense_layers = 2

num_units = [512, 256, 128, 64]
strides = [2, 2, 2, 1, 1]
num_dense_layers = 1
#
#

## Passed ###
num_units = [512, 256, 128, 64]
strides = [2, 2, 1, 1, 1]
num_dense_layers = 2
#
num_units = [512, 256, 128, 64]
strides = [2, 2, 1, 1, 1]
num_dense_layers = 1


initialize_model_train_and_get_features(experiment_name=experiment_name,
                                        z_dim=z_dim,
                                        run_id=run_id,
                                        num_units=num_units,
                                        create_split=create_split,
                                        num_epochs=num_epochs,
                                        root_path=root_path,
                                        learning_rate=learning_rate,
                                        run_evaluation_during_training=True,
                                        eval_interval_in_epochs=1,
                                        model_type="VAE",
                                        strides=strides,
                                        fully_convolutional=False,
                                        num_dense_layers=num_dense_layers,
                                        batch_size=4,
                                        dataset_name="cifar_10",
                                        split_name="split_1"
                                        )
tf.reset_default_graph()
