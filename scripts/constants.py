"""
Constants to perform training and test (it works in multiple tests mode as well).
"""
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_TESTS = 10
TESTS_PER_CASE = 1
NUM_CASES_FIX = 10
NUM_ITERATIONS = 1
VALIDATION = True
FULL_TRAINING = True
VISUALIZATION = "PCA"
VISUALIZATION = "T-SNE"


TYPE_REP = "segments"
#TYPE_REP = "points"

TYPE_LOSS = "log_dist"
#TYPE_LOSS = "dist"

SCHEME_TRAIN = "annealing"
INIT_STRATEGY = "fixed_length_random"

# SCHEME_TRAIN = "pretraining"
#INIT_STRATEGY = "forgy_PCA"
#INIT_STRATEGY = "kmeans++_PCA"
# INIT_STRATEGY = "ksegments"

FIXED_LENGTH_START = 0.50
PERCENTAGE_K = 0.50

# Benchmark datasets
# DATA_SET = "rcv1"
# DATA_SET = "20news"
# DATA_SET = "usps"
# DATA_SET = "mnist"

# Synthetic datasets
#DATA_SET = "synthetic_lines"
#DATA_SET = "synthetic_functions"
DATA_SET = "synthetic_clusters"
TRANSFORMATION = False
NUM_FUNCTIONS = 2

if DATA_SET.find("synthetic") != -1:
    FILLER = DATA_SET.split('_')[1]
else:
    FILLER = DATA_SET

if DATA_SET == "mnist":
    if TYPE_LOSS == "log_dist":
        beta_list = 4*[0.001] + 4*[0.0005] +  4*[0.0001]
    else:
        beta_list = 4*[0.0005] + 4*[0.0001] +  4*[0.00005]
    FULL_TRAINING = False
    gamma_list = 3 * [1e-7, 1e-6, 1e-5, 1e-4]
elif DATA_SET == "usps":
    if TYPE_LOSS == "log_dist":
        beta_list = 4 * [0.001] + 4 * [0.0005] + 4 * [0.0001]
    else:
        beta_list = 4 * [0.001] + 4 * [0.0001] + 4 * [0.00001]

    gamma_list = 3 * [1e-7, 1e-6, 1e-5, 1e-4]
elif DATA_SET == "20news":
    if TYPE_LOSS == "log_dist":
        beta_list = 4*[7e-7] + 4*[8e-7] + 4*[9e-7]
    else:
        beta_list = 4*[7e-7] + 4*[8e-7] + 4*[9e-7]

    gamma_list = 3 * [1e-12, 1e-11, 1e-10, 1e-9]

elif DATA_SET == "rcv1":
    if TYPE_LOSS == "log_dist":
        beta_list = 4 * [1e-8] + 4 * [5e-8] +  4 * [1e-7]
    else:
        beta_list = 4 * [1e-8] + 4 * [5e-8] +  4 * [1e-7]

    gamma_list = 3 * [1e-9, 1e-8, 1e-7, 1e-6]


gamma_list = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
# beta_list = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
beta_list = NUM_CASES_FIX * [0.1]
lambda_ = 0.0
