from generate_config_files import generate_list
from helpers import create_dir
from constants import NUM_TESTS, FILLER
if __name__ == "__main__":

    if FILLER in ["functions","clusters","lines"]:
        filler_aux = "synthetic_" + FILLER
    elif FILLER in ["mnist", "usps", "20news", "rcv1"]:
        filler_aux = "real_" + FILLER

    # create directory for data
    if FILLER in ["functions","clusters","lines"]:
        create_dir("./data/"+filler_aux)
        create_dir("./data/"+filler_aux+"/train")
        create_dir("./data/"+filler_aux+"/test")
        create_dir("./data/"+filler_aux+"/plots")
        # define paths for training and test data and plots
        train_path = "./data/synthetic_"+FILLER+"/train/"
        test_path = "./data/synthetic_"+FILLER+"/test/"
        plot_path = "./data/synthetic_"+FILLER+"/plots/"
        # generate lists of paths
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")
        # create directories
        for i in range(1, NUM_TESTS+1):
            create_dir(list_train_paths[i])
            create_dir(list_test_paths[i])
            create_dir(list_plot_paths[i])

    # define paths for models
    model_path = "./models/"+filler_aux+"/"
    evolution_path = "./models/" + filler_aux + "_evolution/"

    # create directories for models
    create_dir("./models/" + filler_aux)
    create_dir("./models/" + filler_aux + "_evolution")

    # generate lists of paths for models
    list_model_paths = generate_list(model_path, NUM_TESTS, "/")
    list_evolution_paths = generate_list(evolution_path, NUM_TESTS, "/")

    # create directories for models
    for i in range(1, NUM_TESTS+1):
        create_dir(list_model_paths[i])
        create_dir(list_evolution_paths[i])