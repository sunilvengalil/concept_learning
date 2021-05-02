import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features, VAE_FCNN

experiment_name = "Experiment_6"
root_path = "C:/concept_learning_exp/"
z_dim = 5
learning_rate = 0.001
num_epochs = 2
create_split = False
completed_z_dims = 0
run_id = 1
initialize_model_train_and_get_features(experiment_name=experiment_name,
                                        z_dim=z_dim,
                                        run_id=run_id,
                                        create_split=create_split,
                                        num_epochs=num_epochs,
                                        root_path=root_path,
                                        learning_rate=learning_rate,
                                        run_evaluation_during_training=True,
                                        eval_interval_in_epochs=0.25,
                                        model_type=VAE_FCNN
                                        )
tf.reset_default_graph()
