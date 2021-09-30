import argparse
import sys
from logger import Logger
from data_management import load_data_set
from initialization import init_model
from training import train
from testing import test
from helpers import load_config
import numpy as np
import torch

def main():

    ap = argparse.ArgumentParser("kcurves")

    ap.add_argument("mode", choices=["train", "test", "multiple_testing"],
                    help="train or test a model or do multiple testing")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    args = ap.parse_args()

    # set the Logger for the logging
    sys.stdout = Logger(cfg_path = args.config_path)

    if args.mode == "train":
        data_set = load_data_set(cfg_path = args.config_path)
        model = init_model(cfg_path = args.config_path)

        train(cfg_path = args.config_path, model = model, data_set = data_set)

    elif args.mode == "test":
        data_set = load_data_set(cfg_path = args.config_path)
        # init_model is needed since in the training just the state_dict was saved
        model = init_model(cfg_path = args.config_path)
        cfg_file = load_config(args.config_path)
        mode_forced = cfg_file["test"]["mode_forced"]
        _, _, _ = test(cfg_path=args.config_path, model=model, data_set=data_set, mode_forced=mode_forced, mode="final")

    elif args.mode == "multiple_testing":

        # seeds for 10 different runs
        seeds = [9003, 5004, 9671, 3807, 2892, 6832, 6749,  629, 6435, 6637]


        data_set = load_data_set(cfg_path = args.config_path)
        cfg_file = load_config(args.config_path)
        num_iterations = cfg_file["train"]["num_iterations"]
        list_cases = ["k-means on input space", "k-segments on input space", "k-means on latent space",
                      "k-segments on latent space", "model"]
        mat_ACC = np.zeros((num_iterations, 5))
        mat_NMI = np.zeros((num_iterations, 5))
        mat_ARI = np.zeros((num_iterations, 5))
        for iter in range(num_iterations):
            print("Iteration: {}".format(iter+1))
            idx = iter % len(seeds)
            # set the seeds for the initialization of the model
            torch.cuda.manual_seed_all(seeds[idx])
            np.random.seed(seeds[idx])
            model = init_model(cfg_path = args.config_path)
            train(cfg_path = args.config_path, model = model, data_set = data_set)
            list_ACC, list_NMI, list_ARI = test(cfg_path=args.config_path, model=model, data_set=data_set,
                                                mode_forced="test", mode="final")

            # matrices for the metrics
            mat_ACC[iter] = 100 * np.array(list_ACC)
            mat_NMI[iter] = 100 * np.array(list_NMI)
            mat_ARI[iter] = 100 * np.array(list_ARI)

        # get the means and stds for each metric
        mean_ACC = np.mean(mat_ACC, axis = 0)
        std_ACC = np.std(mat_ACC, axis = 0)
        mean_NMI = np.mean(mat_NMI, axis = 0)
        std_NMI = np.std(mat_NMI, axis = 0)
        mean_ARI = np.mean(mat_ARI, axis = 0)
        std_ARI = np.std(mat_ARI, axis = 0)

        # trace the metrics
        print("List of accuracies: ")
        print(mat_ACC[:, 4])
        for i in range(5):
            print("Case {}: {}".format(i+1, list_cases[i]))
            print("\tAvg ACC = {}".format(np.round(mean_ACC[i],2)))
            print("\tAvg NMI = {}".format(np.round(mean_NMI[i],2)))
            print("\tAvg ARI = {}".format(np.round(mean_ARI[i],2)))
            print("\tStd ACC = {}".format(np.round(std_ACC[i],2)))
            print("\tStd NMI = {}".format(np.round(std_NMI[i],2)))
            print("\tStd ARI = {}".format(np.round(std_ARI[i],2)))
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()