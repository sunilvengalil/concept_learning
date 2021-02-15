import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features

experiment_name = "Experiment_1"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim_range = [5, 15, 2]
learning_rate = 0.001
num_epochs = 10
num_runs = 5
create_split = False
completed_z_dims = 0

for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
    for run_id in range(num_runs):
        initialize_model_train_and_get_features(experiment_name=experiment_name,
                                                z_dim=z_dim,
                                                run_id=run_id,
                                                create_split=create_split,
                                                num_epochs=num_epochs,
                                                root_path=root_path,
                                                learning_rate=learning_rate,
                                                run_evaluation_during_training=True,
                                                eval_interval_in_epochs=0.25,
                                                model_type="VAE")
        tf.reset_default_graph()
