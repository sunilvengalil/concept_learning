import unittest
from clearn.config import ExperimentConfig, get_base_path
from clearn.utils.annotation_utils import combine_annotation_sessions, combine_multiple_annotations


class ResolveDuplicate(unittest.TestCase):
    keys = ["manual_annotation_set_1", "manual_annotation_set_2"]
    max_epoch = 5

    N_3 = 32
    N_2 = 128
    N_1 = 64
    z_dim = 10
    run_id = 1
    root_path = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
    exp_config = ExperimentConfig(root_path,
                                  4,
                                  z_dim,
                                  [N_1, N_2, N_3],
                                  num_cluster_config=None
                                  )
    #
    # def test_resolve_duplicates(self):
    #
    #     N_3 = 32
    #     N_2 = 128
    #     N_1 = 64
    #     z_dim = 10
    #     run_id = 1
    #     root_path = "/Users/sunilkumar/concept_learning_old/image_classification_old/"
    #     exp_config = ExperimentConfig(self.root_path,
    #                                   4,
    #                                   self.z_dim,
    #                                   [self.N_1, self.N_2, self.N_3],
    #                                   num_cluster_config=None
    #                                   )
    #     exp_config.check_and_create_directories(run_id)
    #
    #     # Read all the individual data frames into a dictionary of format {"annotator_id"}
    #     data_dict = combine_annotation_sessions(keys=self.keys,
    #                                             exp_config=exp_config,
    #                                             run_id=run_id,
    #                                             max_epoch=self.max_epoch)
    #
    #     df = data_dict["manual_annotation_set_1"]["data_frame"]
    #     df = df[df["has_multiple_value"]]
    #     print(df.shape)
    #     self.assertEquals(df.shape[0], 0 )
    #
    #     df = data_dict["manual_annotation_set_2"]["data_frame"]
    #     df = df[df["has_multiple_value"]]
    #     print(df.shape)
    #     self.assertEquals(df.shape[0], 9)

    def test_combine_multiple_annotations(self):
        run_id = 1
        exp_config = ExperimentConfig(self.root_path,
                                      4,
                                      self.z_dim,
                                      [self.N_1, self.N_2, self.N_3],
                                      num_cluster_config=None
                                      )
        exp_config.check_and_create_directories(run_id)
        # Read all the individual data frames into a dictionary of format {"annotator_id"}
        base_path = get_base_path(exp_config.root_path,
                                  exp_config.Z_DIM,
                                  exp_config.num_units[2],
                                  exp_config.num_units[1],
                                  exp_config.num_cluster_config,
                                  run_id=run_id
                                  )

        data_dict = combine_annotation_sessions(keys=self.keys,
                                                base_path=base_path,
                                                max_epoch=self.max_epoch)

        df_set_1 = data_dict["manual_annotation_set_1"]["data_frame"]
        df_set_1 = df_set_1[df_set_1["has_multiple_value"]]
        self.assertEqual(df_set_1.shape[0], 0)

        df_set_2 = data_dict["manual_annotation_set_2"]["data_frame"]
        df_set_2 = df_set_2[df_set_2["has_multiple_value"]]
        self.assertEqual(df_set_2.shape[0], 9)

        combine_multiple_annotations(data_dict, exp_config, run_id)

        df_set_1 = data_dict["manual_annotation_set_1"]["data_frame"]
        df_set_1 = df_set_1[df_set_1["has_multiple_value"]]
        self.assertEqual(df_set_1.shape[0], 0)

        df_set_2 = data_dict["manual_annotation_set_2"]["data_frame"]
        df_set_2 = df_set_2[df_set_2["has_multiple_value"]]
        self.assertEqual(df_set_2.shape[0], 9)


if __name__ == '__main__':
    unittest.main()
