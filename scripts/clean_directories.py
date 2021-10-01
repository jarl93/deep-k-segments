from generate_config_files import generate_list
from helpers import create_dir
import os
import shutil
import argparse
from constants import NUM_TESTS, FILLER

def delete_dir(path):
    """
    Performs rm -r to a given path.
    Arguments:
        path: path with the file or directory to be removed.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)

    return None

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--all', action='store_true',
                    help='Clean all the directories including the config files.')

    ap.add_argument('-cfg', '--configs_generation', action='store_true',
                    help='Clean the config files for the generation process.')

    ap.add_argument('-cft', '--configs_training', action='store_true',
                    help='Clean the config files for training and testing.')

    ap.add_argument('-e', '--evolution', action='store_true',
                    help='Clean the directories of the models and graphs of the evolution of the network.')

    ap.add_argument('-m', '--models', action='store_true',
                    help='Clean the directories of the models and its respective plots.')

    ap.add_argument('-p', '--plots', action='store_true',
                    help='Clean the plots of the data generated.')

    ap.add_argument('-d', '--data', action='store_true',
                    help='Clean the training and test data.')

    args = ap.parse_args()
    if FILLER in ["functions","clusters","lines"]:
        filler_aux = "synthetic_" + FILLER
    elif FILLER in ["mnist", "usps", "20news", "rcv1"]:
        filler_aux = "real_" + FILLER

    config_path ="./configs/"+filler_aux+"/"
    config_generation_path = "./configs/"+filler_aux+"_generation/"

    train_path = "./data/"+filler_aux+"/train/"
    test_path = "./data/"+filler_aux+"/test/"
    plot_path = "./data/"+filler_aux+"/plots/"
    model_path = "./models/"+filler_aux+"/"
    evolution_path = "./models/" + filler_aux + "_evolution/"

    if args.configs_generation or args.all:
        delete_dir(config_generation_path)
    if args.configs_training or args.all:
        delete_dir(config_path)

    list_train_paths = generate_list(train_path, NUM_TESTS, "/")
    list_test_paths = generate_list(test_path, NUM_TESTS, "/")
    list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")
    list_model_paths = generate_list(model_path, NUM_TESTS, "/")
    list_evolution_paths = generate_list(evolution_path, NUM_TESTS, "/")

    for i in range(1, NUM_TESTS+1):
        if args.data or args.all and FILLER in ["functions","clusters","lines"]:
            delete_dir(list_train_paths[i])
            delete_dir(list_test_paths[i])
            create_dir(list_train_paths[i])
            create_dir(list_test_paths[i])
        if args.plots or args.all and FILLER in ["functions","clusters","lines"]:
            delete_dir(list_plot_paths[i])
            create_dir(list_plot_paths[i])
        if args.models or args.all:
            delete_dir(list_model_paths[i])
            create_dir(list_model_paths[i])
        if args.evolution or args.all:
            delete_dir(list_evolution_paths[i])
            create_dir(list_evolution_paths[i])


if __name__ == "__main__":
    main()