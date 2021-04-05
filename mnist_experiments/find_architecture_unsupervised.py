import tensorflow as tf
from clearn.experiments.experiment import initialize_model_train_and_get_features, MODEL_TYPE_VAE_UNSUPERVISED

experiment_name = "find_architecture_unsup"
root_path = "/Users/sunilv/concept_learning_exp/"
z_dim_range = [5, 30, 2]
learning_rate = 0.001
num_epochs = 50
create_split = False
run_id = 1
num_unit_list = [[64, 32],
                      [32, 32],
                      [16, 32],
                      [8, 32],
                      [4, 32],
                      [2, 32],
                      ]
for num_unit in num_unit_list:
    for z_dim in range(z_dim_range[0], z_dim_range[1], z_dim_range[2]):
        initialize_model_train_and_get_features(experiment_name=experiment_name,
                                                num_units=num_unit,
                                                batch_size=128,
                                                z_dim=z_dim,
                                                run_id=run_id,
                                                create_split=create_split,
                                                num_epochs=num_epochs,
                                                root_path=root_path,
                                                learning_rate=learning_rate,
                                                model_save_interval=10,
                                                run_evaluation_during_training=True,
                                                eval_interval_in_epochs=1,
                                                model_type=MODEL_TYPE_VAE_UNSUPERVISED,
                                                return_latent_vector=False,
                                                write_predictions=False
                                                )
        tf.reset_default_graph()
