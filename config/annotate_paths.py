import os
from config.common_path import ROOT_PATH, BATCH_SIZE, SPLIT_NAME, DATASET_NAME, MODEL_NAME
Z_DIM = 10
N_3 = 16
N_2 = 128
BASE_PATH = os.path.join(ROOT_PATH, "Exp_{:02d}_{:03}_{:03d}/".format(Z_DIM, N_3,N_2))

# PREDICTION_RESULTS_PATH = BASE_PATH+"prediction_results/"
DATASET_ROOT_PATH = os.path.join(BASE_PATH, DATASET_NAME + "/")
DATASET_PATH = os.path.join(DATASET_ROOT_PATH, SPLIT_NAME+"/")

DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS = os.path.join(ROOT_PATH, "datasets/" + DATASET_NAME)
MODEL_NAME_WITH_CONFIG = "{}_{}_{:2d}_{:02d}".format(MODEL_NAME, DATASET_NAME, BATCH_SIZE, Z_DIM)

# Possible values "CORRECT"/WRONG
PRESS_ENTER_FOR = "CORRECT"


