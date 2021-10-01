# libraries
import numpy as np
from helpers import load_config, Read_Two_Column_File, Read_One_Column_File
from data import SyntheticDataset
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data_set(cfg_path, verbose = True):
    """
    Loads and prepare a data set specified in the config file (with path cfg_path).
    Arguments:
        cfg_path: config path of the config file.
    Output:
        data_set: data set loaded with training and test set.

    """
    # load config file
    cfg_file = load_config(cfg_path)

    if verbose:
        print("Loading dataset {}...".format(cfg_file["data"]["data_set"]))


    # Pre-process the datasets as in the paper of deep k-means by Fard et. al.
    # Implementation taken from https://github.com/MaziarMF/deep-k-means.
    # Case for benchmark datasets
    if cfg_file["data"]["data_set"] == "mnist" or cfg_file["data"]["data_set"] == "usps" or \
            cfg_file["data"]["data_set"] == "20news" or cfg_file["data"]["data_set"] == "rcv1":


        if cfg_file["data"]["data_set"] == "mnist":
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            # scale data
            X = mnist.data / 255.0
            Y = mnist.target.astype(np.int64)
        elif cfg_file["data"]["data_set"] == "usps":
            usps = fetch_openml(name = 'USPS', version = 2)
            X = usps.data
            Y = usps.target.astype(np.int64) - 1
        elif cfg_file["data"]["data_set"] == "20news":
            _20news = fetch_20newsgroups(subset="all")
            X = _20news.data
            Y = _20news.target
            vectorizer = TfidfVectorizer(max_features=2000)
            X = vectorizer.fit_transform(X) # Keep only the 2000 top words in the vocabulary
            X = X.toarray() # Switch from sparse matrix to full matrix
        elif cfg_file["data"]["data_set"] == "rcv1":

            rcv1 = fetch_rcv1(subset="all")
            data = rcv1.data
            target = rcv1.target
            # Get the split between training/test set and validation set
            idx_test = Read_One_Column_File("./split/rcv1/test", 'int')
            n_test = idx_test.shape[0]
            idx_val = Read_One_Column_File("./split/rcv1/validation", 'int')
            n_validation = idx_val.shape[0]

            # Filter the dataset
            # Keep only the data points in the test and validation sets
            test_data = data[idx_test]
            test_target = target[idx_test]
            validation_data = data[idx_val]
            validation_target = target[idx_val]
            data = sp.vstack([test_data, validation_data])
            target = sp.vstack([test_target, validation_target])


            # Pre-process the dataset
            # Filter words based on tf-idf
            # Sum of tf-idf for all words based on the filtered dataset
            sum_tfidf = np.asarray(sp.spmatrix.sum(data, axis=0))[0]
            word_indices = np.argpartition(-sum_tfidf, 2000)[:2000] # Keep only the 2000 top words in the vocabulary
            data = data[:, word_indices].toarray() # Switch from sparse matrix to full matrix
            ## Retrieve the unique label (corresponding to one of the specified categories) from target's label vector
            names = rcv1.target_names
            category_names = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
            category_indices = [i for i in range(len(names)) if names[i] in category_names]
            # To rescale the indices between 0 and some K
            dict_category_indices = {j: i for i, j in enumerate(category_indices)}
            filtered_target = []
            for i in range(target.shape[0]): # Loop over data points
                target_coo = target[i].tocoo().col
                filtered_target_coo = [t for t in target_coo if t in category_indices]
                assert len(filtered_target_coo) == 1 # Only one relevant label per document because of pre-filtering
                filtered_target.append(dict_category_indices[filtered_target_coo[0]])
            target = np.asarray(filtered_target)

            X = data
            Y = target


        filler_path = cfg_file["data"]["data_set"]
        full_training = cfg_file["data"]["full_training"]
        validation = cfg_file["data"]["validation"]

        if full_training:
            X_train = X
            Y_train = Y
        else:
            if cfg_file["data"]["data_set"] == "mnist":
                idx_train = Read_One_Column_File('./split/'+filler_path+'/train_20', 'int')
                X_train = X[idx_train]
                Y_train = Y[idx_train]
            else:
                raise ValueError("Partial training applies only to MNIST!")

        if cfg_file["data"]["data_set"] == "rcv1":
            # Update test_indices and validation_indices to fit the new data indexing
            if validation:
                idx_test = np.asarray(range(n_test, n_test + n_validation))
            else:
                idx_test = np.asarray(range(0, n_test))
        else:
            if validation:
                idx_test = Read_One_Column_File('./split/'+filler_path+'/validation', 'int')
            else:
                idx_test = Read_One_Column_File('./split/'+filler_path+'/test', 'int')

        X_test = X[idx_test]
        Y_test = Y[idx_test]

        # Init the cluster centers at zero for simplicity
        num_classes = cfg_file["data"]["num_classes"]
        input_dim = cfg_file["model"]["input_dim"]
        centers_train = np.zeros((num_classes, input_dim))
        centers_test  = np.zeros((num_classes, input_dim))


    # Case for synthetic datasets
    elif cfg_file["data"]["data_set"] in ["synthetic_functions", "synthetic_clusters", "synthetic_lines"]:

        # load the generated data
        X_train = np.load(cfg_file["data"]["train"]+"X_train.npy")
        Y_train = np.load(cfg_file["data"]["train"]+"Y_train.npy")
        Y_train = Y_train.astype(np.int64)
        centers_train = np.load(cfg_file["data"]["train"]+"centers_train.npy")
        X_test =  np.load(cfg_file["data"]["test"]+"X_test.npy")
        Y_test = np.load(cfg_file["data"]["test"]+ "Y_test.npy")
        Y_test = Y_test.astype(np.int64)
        centers_test = np.load(cfg_file["data"]["test"]+"centers_test.npy")

    elif cfg_file["data"]["data_set"] == "basic_benchmarking":
        X_train = Read_Two_Column_File(cfg_file["data"]["train_data"], 'float')
        centers_train = Read_Two_Column_File(cfg_file["data"]["train_centers"], 'float')
        X_train, centers_train = normalize_data(X_train, centers_train)
        Y_train = Read_One_Column_File(cfg_file["data"]["train_labels"], 'float')
        # use labels starting at 0
        Y_train = Y_train - 1

        # In this case the train and test set are the same
        X_test = X_train
        Y_test = Y_train
        centers_test = centers_train



    # common code for all the data sets
    # save the matrices with the training and test data
    np.save(cfg_file["data"]["train"] + "X_train", X_train)
    np.save(cfg_file["data"]["train"] + "Y_train", Y_train)
    np.save(cfg_file["data"]["train"]+ "centers_train", centers_train)
    np.save(cfg_file["data"]["test"] + "X_test", X_test)
    np.save(cfg_file["data"]["test"] + "Y_test", Y_test)
    np.save(cfg_file["data"]["test"] + "centers_test", centers_test)

    # trace the shapes of the train and test data sets
    if verbose:
        print("Shape X_train: ", X_train.shape)
        print("Shape Y_train: ", Y_train.shape)
        print("Shape X_test: ", X_test.shape)
        print("Shape Y_test: ", Y_test.shape)

    # create the data set with the help of the SyntheticDataset class
    train_dataset = SyntheticDataset(data = X_train, labels = Y_train)
    test_dataset = SyntheticDataset(data = X_test, labels = Y_test)

    data_set = (train_dataset, test_dataset)

    if verbose:
        print("Dataset loaded!")

    return data_set

def normalize_data(X, centers = None, verbose = False):
    """
    Normalizes the data by subtracting the mean and dividing by the variance.
    Arguments:
        X: numpy array with the data to apply normalization.
        centers: cluster centers (if any) to apply normalization.
        verbose: bool variable to print out sanity checks.

    Output:
        X:  data normalized.
        centers (if any): cluster centers normalized.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    if verbose:
        print("Function: ", normalize_data.__name__)
        print("Mean: ", mean)
        print("Std: ", std)

    X = (X - mean) / std
    if centers is not None:
        centers = (centers - mean) / std
        return X, centers
    else:
        return X
def scale_data(X, scale_factor, verbose = False):
    """
    Scales data, such that all the points are in the square [(-1, -1), (1,1)]
    bottom_left = (-1,-1), upper_right = (1,1).
    Arguments:
        X: numpy array with the data to apply rescaling.
        scale_factor: factor to scale all the values.
        verbose: boolean variable to print sanity checks.
    Outputs:
        X_scaled: numpy array with the data scaled.
    """
    X_abs = np.absolute(X)
    x_max = np.max(X_abs)
    X_scaled = scale_factor * X / x_max

    return X_scaled

def split_data_loader(data_loader):
    """
    Splits the data loader into data (X) and labels(Y).
    Arguments:
        data_loader: data loader.
    Outputs:
        X: data.
        Y: true labels.
    """
    X = None  # array to store data
    Y = None  # array to store labels
    for batch_idx, data in enumerate(data_loader):
        x, y = data
        x = x.reshape(x.shape[0], -1)
        if batch_idx == 0:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.hstack((Y, y))

    return X, Y