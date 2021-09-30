from helpers import create_dir
import numpy as np
import argparse
from constants import DEVICE, NUM_TESTS, NUM_CASES_FIX, NUM_ITERATIONS, DATA_SET, FILLER, SCHEME_TRAIN, INIT_STRATEGY, \
VALIDATION, NUM_FUNCTIONS, TESTS_PER_CASE, TRANSFORMATION, beta_list, gamma_list, lambda_, \
TYPE_REP, TYPE_LOSS, FIXED_LENGTH_START, PERCENTAGE_K, FULL_TRAINING, VISUALIZATION

def generate_list(str_, num_tests, char):
    """
    Add documentation.
    :param str_:
    :param num_tests:
    :param char:
    :return:
    """
    list_str = [None]
    for i in range(1, num_tests+1):
        num = ""
        # put 0's at the beginning w.r.t. num_tests
        dif = int(np.log10(num_tests)) - int(np.log10(i))
        for j in range(dif):
            num += "0"
        # add the number given by the iterator
        num += str(i)
        # handle cases depending on the char
        if char == "_":
            list_str.append(str_ + "_" + num)
        else:
            list_str.append(str_ + num + "/")
    return list_str
def write_config_training(path_config_files, list_names, list_model_names, list_model_paths,
                          list_evolution_paths, list_log_names):
    """
    Writes the config file for the training and testing.
    :param path_config_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_model_names:
    :param list_model_paths:
    :param list_evolution_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # in the case for real data, the directory does not change
    filler_aux = DATA_SET


    # data
    if DATA_SET.find("synthetic") == -1:
        train_path = "./data/"+filler_aux+"/"
        test_path = "./data/"+filler_aux+"/"
    else:
        train_path = "./data/"+filler_aux+"/train/01/"
        test_path = "./data/"+filler_aux+"/test/01/"

    # training
    batch_size = 256
    lr = 0.001
    batch_frequency_loss = 1
    # test:
    mode_forced = "test"

    if SCHEME_TRAIN == "annealing":
        num_epochs = 200
        epochs_warmup = 0
        alpha_type_list = NUM_CASES_FIX * ["annealing"]
        alpha_list = NUM_CASES_FIX * [0.1]
        annealing_frequency_change = 5
    elif SCHEME_TRAIN == "pretraining":
        num_epochs = 150
        epochs_warmup = 50
        alpha_type_list = NUM_CASES_FIX * ["fixed"]
        alpha_list = NUM_CASES_FIX * [1000]
        annealing_frequency_change = 1

    # hyperparameters for loss of distance and length of segment
    beta_type_list = NUM_CASES_FIX * ["fixed"]
    gamma_type_list = NUM_CASES_FIX * ["fixed"]

    # type of representatives and type of loss
    type_rep_list = NUM_CASES_FIX * [TYPE_REP]
    type_loss_list = NUM_CASES_FIX * [TYPE_LOSS]


    # initialization
    min_init = -1
    diff_init = 2
    percentage_K_list = NUM_CASES_FIX *  [PERCENTAGE_K]
    init_strategy_list = NUM_CASES_FIX * [INIT_STRATEGY]
    space_init = "latent"
    #space_init = "input"

    # architecture for the autoencoder depending on the dataset:
    if DATA_SET.find("synthetic") == -1:
        if DATA_SET == "mnist":
            input_dim = 784
            num_classes = 10
            latent_dim = 10
        elif DATA_SET == "usps":
            input_dim = 256
            num_classes = 10
            latent_dim = 10
        elif DATA_SET == "20news":
            input_dim = 2000
            num_classes = 20
            latent_dim = 20
        elif DATA_SET == "rcv1":
            input_dim = 2000
            num_classes = 4
            latent_dim = 4
        layer_sizes_encoder = [input_dim, 500, 500, 2000]
        layer_sizes_decoder = [2000, 500, 500, input_dim]
    else:
        latent_dim = 2
        if TRANSFORMATION:
            input_dim = 100
            layer_sizes_encoder = [input_dim, 64, 32, 16]
            layer_sizes_decoder = [16, 32, 64, input_dim]
        else:
            input_dim = 2
            layer_sizes_encoder = [input_dim]
            layer_sizes_decoder = [latent_dim]
            if DATA_SET == "synthetic_clusters" or DATA_SET == "synthetic_functions":
                num_classes = 2
            elif DATA_SET == "synthetic_lines":
                num_classes = 9

    # tracing and visualization:
    evolution = True
    save_evolution = True
    epochs_frequency_evolution = 20
    batch_frequency = 10
    num_points_inter = 10
    ######################
    x_interval = [-1, 1]
    y_interval = [-1, 1]
    delta_interval = 0.01
    levels_contour = 20

    for i in range(1, NUM_TESTS + 1):
        idx = (i -1) // TESTS_PER_CASE
        type_loss = type_loss_list[idx]
        type_rep = type_rep_list[idx]
        beta_init = beta_list[idx]
        beta_type = beta_type_list[idx]
        gamma_init = gamma_list[idx]
        gamma_type = gamma_type_list[idx]
        percentage_K = percentage_K_list[idx]
        init_strategy = init_strategy_list[idx]
        alpha_init = alpha_list[idx]
        alpha_type = alpha_type_list[idx]
        path = path_config_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the files
        f.write("name: " + list_names[i] + "\n")
        f.write("data:\n")
        f.write("  data_set: " + str(DATA_SET) + "\n")
        f.write("  train: " + str(train_path) + "\n")
        f.write("  test: " + str(test_path) + "\n")
        f.write("  validation: " + str(VALIDATION) + "\n")
        f.write("  full_training: " + str(FULL_TRAINING) + "\n")
        f.write("  num_classes: " + str(num_classes) + "\n")

        f.write("train:\n")
        f.write("  num_iterations: " + str(NUM_ITERATIONS) + "\n")
        f.write("  batch_size: " + str(batch_size) + "\n")
        f.write("  num_epochs: " + str(num_epochs) + "\n")
        f.write("  epochs_warmup: " + str(epochs_warmup) + "\n")
        f.write("  annealing_frequency_change: " + str(annealing_frequency_change) + "\n")
        f.write("  alpha_type: " + str(alpha_type) + "\n")
        f.write("  alpha_init: " + str(alpha_init) + "\n")
        f.write("  beta_type: "+str(beta_type)+ "\n")
        f.write("  beta_init: "+ format(beta_init, ".9f") + "\n")
        f.write("  gamma_type: " + str(gamma_type) + "\n")
        f.write("  gamma_init: " + format(gamma_init, ".12f") + "\n")
        f.write("  lambda: " + format(lambda_, ".6f") + "\n")
        f.write("  init_strategy: " + str(init_strategy) + "\n")
        f.write("  min_init: " + str(min_init) + "\n")
        f.write("  diff_init: " + str(diff_init) + "\n")
        f.write("  space_init: " + str(space_init) + "\n")
        f.write("  scheme_train: " + str(SCHEME_TRAIN) + "\n")
        f.write("  percentage_K: " + str(percentage_K) + "\n")
        f.write("  type_loss: " + str(type_loss) + "\n")
        f.write("  type_rep: " + str(type_rep) + "\n")
        f.write("  fixed_length_start: " + str(FIXED_LENGTH_START) + "\n")
        f.write("  lr: " + str(lr) + "\n")
        f.write("  batch_frequency_loss: " + str(batch_frequency_loss) + "\n")
        f.write("  evolution : " + str(evolution) + "\n")
        f.write("  epochs_frequency_evolution : " + str(epochs_frequency_evolution) + "\n")
        f.write("  save_evolution : " + str(save_evolution) + "\n")

        f.write("test:\n")
        f.write("  mode_forced: " + str(mode_forced) + "\n")
        f.write("  batch_size: " + str(batch_size) + "\n")

        f.write("model:\n")
        f.write("  path: " + list_model_paths[i] + "\n")
        f.write("  evolution_path: " + list_evolution_paths[i]+ "\n")
        f.write("  name: " + list_model_names[i] + "\n")
        f.write("  save: True\n")
        f.write("  device: " + str(DEVICE) + "\n")
        f.write("  encoder:\n")
        f.write("    layer_sizes: " + str(layer_sizes_encoder) + "\n")
        f.write("    last_nn_layer: Identity\n")
        f.write("  decoder:\n")
        f.write("    layer_sizes: " + str(layer_sizes_decoder) + "\n")
        f.write("    last_nn_layer: Identity\n")
        f.write("  input_dim: " + str(input_dim) + "\n")
        f.write("  latent_dim: " + str(latent_dim) + "\n")

        f.write("tracing:\n")
        f.write("  log_name: "+list_log_names[i]+"\n")
        f.write("  show_images: False\n")
        f.write("  images_to_show: 10\n")
        f.write("  visualize_latent: True\n")
        f.write("  x_interval: " + str(x_interval) + "\n")
        f.write("  y_interval: " + str(y_interval) + "\n")
        f.write("  delta_interval: " + str(delta_interval) + "\n")
        f.write("  levels_contour: " + str(levels_contour) + "\n")
        f.write("  batch_frequency: " + str(batch_frequency) + "\n")
        f.write("  num_points_inter: " + str(num_points_inter) + "\n")
        f.write("  visualization: " + str(VISUALIZATION) + "\n")
        
        f.close()

    return None

def write_config_synthetic_clusters_generation(path_config_generation_files, list_names, list_train_paths,
                                                list_test_paths, list_plot_paths):
    """
    Writes config generation files for the synthetic clusters data set.
    :param path_config_generation_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_plot_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # TODO: Consider to make variables for the boolean hyperparameters
    # also equal to number of classes
    num_centers = 2
    dim = 2
    center_box = [-100, 100]
    cluster_std = 5
    num_samples_train = 5000
    num_samples_test = 1000
    normalize = True
    for i in range(1, NUM_TESTS + 1):
        random_state = (i -1) % TESTS_PER_CASE
        path = path_config_generation_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the files
        f.write("name: " + list_names[i] + "\n")
        f.write("clusters:\n")
        f.write("  num_centers: " + str(num_centers) + "\n")
        f.write("  center_box: " + str(center_box) + "\n")
        f.write("  cluster_std: " + str(cluster_std) + "\n")
        f.write("  random_state: " + str(random_state) + "\n")
        f.write("  dim: " + str(dim) + "\n")
        f.write("data:\n")
        f.write("  save: True\n")
        f.write("  plot: True\n")
        f.write("  normalize: "+ str(normalize) + "\n")
        f.write("  train:\n")
        f.write("    path: " + list_train_paths[i] + "\n")
        f.write("    num_samples: " + str(num_samples_train) + "\n")
        f.write("  test:\n")
        f.write("    path: " + list_test_paths[i] + "\n")
        f.write("    num_samples: " + str(num_samples_test) + "\n")
        f.write("  plots:\n")
        f.write("    path: " + list_plot_paths[i] + "\n")
        f.close()

    return None

def write_config_synthetic_lines_generation(path_config_generation_files, list_names, list_train_paths,
                                                list_test_paths, list_plot_paths):
    """
    Writes config generation files for the synthetic clusters data set.
    :param path_config_generation_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_plot_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # TODO: Consider to make variables for the boolean hyperparameters
    # also equal to number of classes
    num_samples_train = 5000
    num_samples_test = 1000
    normalize = True
    transformation = False
    rotation = True

    # non-linear transformation
    non_linear = "sigmoid"
    list_dimensions = [[10, 2], [50, 10], [100, 50]]
    for i in range(1, NUM_TESTS + 1):
        path = path_config_generation_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the files
        f.write("data:\n")
        f.write("  save: True\n")
        f.write("  plot: False\n")
        f.write("  normalize: " + str(normalize) + "\n")
        f.write("  transformation: " + str(transformation) + "\n")
        f.write("  train:\n")
        f.write("    path: " + list_train_paths[i] + "\n")
        f.write("    num_samples: " + str(num_samples_train) + "\n")
        f.write("    rotation: " + str(rotation) + "\n")
        f.write("  test:\n")
        f.write("    path: " + list_test_paths[i] + "\n")
        f.write("    num_samples: " + str(num_samples_test) + "\n")
        f.write("  plots:\n")
        f.write("    path: " + list_plot_paths[i] + "\n")
        # define transformation
        f.write("transformation:\n")
        f.write("  non_linear: " + str(non_linear) + "\n")
        f.write("  list_dimensions:\n")
        for dimension in list_dimensions:
            f.write("    - " + str(dimension) + "\n")
        f.close()

    return None

def write_config_synthetic_functions_generation(path_config_generation_files, list_names, list_train_paths,
                                                list_test_paths, list_plot_paths):
    """
    Writes config generation files for the synthetic functions data set.
    :param path_config_generation_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_plot_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # TODO: Consider to make variables for the boolean hyperparameters
    # list of hyperparameters for functions
    names_F = ["F1", "F2"]

    # paramters for case x < y
    # amp = [10, 10]
    # frec = [0.1, 0.1]
    # interval = [[0, 10], [0,10]]
    # # shift = [0, 30]
    # d_shift = 50

    # paramters for case linear separable
    amp = [3, 3]
    frec = [0.01, 0.01]
    interval = [[0, 100], [0, 100]]
    # shift = [0, 30]
    d_shift = 30

    char_to_plot = ['x', 'o']
    color_to_plot = ['red', 'green']
    # class distribution: 50-50
    train_num_samples = [3000, 3000]
    test_num_samples = [500, 500]


    # # class distribution: 70-30
    # train_num_samples = [4200, 1800]
    # test_num_samples = [700, 300]


    # # class distribution: 90-10
    # train_num_samples = [5400, 600]
    # test_num_samples = [900, 100]

    normalize = True
    non_linear = "sigmoid"
    list_dimensions = [[10, 2], [50, 10], [100, 50]]
    transformation = False

    shift_list = []

    for i in range(1, NUM_TESTS + 1):
        path = path_config_generation_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the file
        f.write("name: " + list_names[i] + "\n")

        # code to generate different cases
        if i <= TESTS_PER_CASE:
            # enforce a separation of at least d_shift units between the two functions
            # shift_1 = np.random.randint(-50, 50)
            # shift_2 = np.random.randint(-50, 50)
            # while abs(shift_1 - shift_2) < d_shift:
            #     shift_1 = np.random.randint(-50, 50)
            #     shift_2 = np.random.randint(-50, 50)

            # fixed case
            shift_1 = -40
            shift_2 = 10

            shift = [shift_1, shift_2]
            shift_list.append(shift)
        else:
            idx = (i-1) % TESTS_PER_CASE
            shift = shift_list[idx]

        # functions
        for j in range(NUM_FUNCTIONS):
            f.write(names_F[j]+":\n")
            f.write("  amp: " + str(amp[j]) + "\n")
            f.write("  frec: " + str(frec[j]) + "\n")
            f.write("  interval: " + str(interval[j]) + "\n")
            f.write("  shift: " + str(shift[j]) + "\n")
            f.write("  char_to_plot: " + str(char_to_plot[j]) + "\n")
            f.write("  color_to_plot: " + str(color_to_plot[j]) + "\n")
            f.write("  train_num_samples: " + str(train_num_samples[j]) + "\n")
            f.write("  test_num_samples: " + str(test_num_samples[j]) + "\n")

        # data parameters for generation
        f.write("data:\n")
        f.write("  save: True\n")
        f.write("  plot: True\n")

        f.write("  normalize: " + str(normalize) + "\n")
        f.write("  transformation: " + str(transformation) + "\n")
        f.write("  train:\n")
        f.write("    path: " + list_train_paths[i] + "\n")
        f.write("  test:\n")
        f.write("    path: " + list_test_paths[i] + "\n")
        f.write("  plots:\n")
        f.write("    path: " + list_plot_paths[i] + "\n")

        # define transformation
        f.write("transformation:\n")
        f.write("  non_linear: " + str(non_linear) + "\n")
        f.write("  list_dimensions:\n")
        for dimension in list_dimensions:
            f.write("    - "+str(dimension)+"\n")

        f.close()

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--all', action='store_true',
                    help='Generate config files for the generation process and for training (and testing).')

    ap.add_argument('-g', '--generation', action='store_true',
                    help='Generate config files for the generation process.')

    ap.add_argument('-t', '--training', action='store_true',
                    help='Generate config files for training and testing.')

    args = ap.parse_args()

    if FILLER in ["functions","clusters","lines"]:
        filler_aux = "synthetic_" + FILLER
    elif FILLER in ["mnist", "usps", "20news", "rcv1"]:
        filler_aux = "real_" + FILLER

    # create directory for the config files
    path_config_files = "./configs/"+filler_aux+"/"
    create_dir(path_config_files)

    # main paths for training data, test data and plots (generated data)
    train_path = "./data/"+filler_aux+"/train/"
    test_path = "./data/"+filler_aux+"/test/"
    plot_path = "./data/"+filler_aux+"/plots/"

    # code to generate config files that generate data
    if args.generation or args.all:
        # create directory for the config generation files

        if FILLER in ["functions", "clusters", "lines"]:
            path_config_generation_files = "./configs/"+filler_aux+"_generation/"
            create_dir(path_config_generation_files)
            name = filler_aux+"_generation"
            # lists with the paths for the training data, test data and plots
            list_names = generate_list(name, NUM_TESTS, "_")
            list_train_paths = generate_list(train_path, NUM_TESTS, "/")
            list_test_paths = generate_list(test_path, NUM_TESTS, "/")
            list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")

            # write config file for generation
            if DATA_SET == "synthetic_functions":
                write_config_synthetic_functions_generation(path_config_generation_files, list_names, list_train_paths,
                                                            list_test_paths, list_plot_paths)
            elif DATA_SET == "synthetic_clusters":
                write_config_synthetic_clusters_generation(path_config_generation_files, list_names, list_train_paths,
                                                           list_test_paths, list_plot_paths)
            elif DATA_SET == "synthetic_lines":
                write_config_synthetic_lines_generation(path_config_generation_files, list_names, list_train_paths,
                                                           list_test_paths, list_plot_paths)


    # code to generate config files for training and testing
    if args.training or args.all:

        # create directories for config files and define path


        path_config_files = "./configs/"+filler_aux+"/"
        create_dir(path_config_files)
        name = filler_aux
        model_path = "./models/"+filler_aux+"/"
        evolution_path = "./models/"+filler_aux+"_evolution/"

        model_name = "model_"+filler_aux
        log_name = filler_aux

        # create lists with train, test and model paths
        list_names = generate_list(name, NUM_TESTS, "_")
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        list_model_names = generate_list(model_name, NUM_TESTS, "_")
        list_log_names = generate_list(log_name, NUM_TESTS, "_")
        list_model_paths = generate_list(model_path, NUM_TESTS, "/")
        list_evolution_paths = generate_list(evolution_path, NUM_TESTS, "/")

        # write config file for training
        write_config_training(path_config_files, list_names,  list_model_names,
                              list_model_paths, list_evolution_paths, list_log_names)

if __name__ == "__main__":
    main()
