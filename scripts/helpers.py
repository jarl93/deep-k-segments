import os
def create_dir(path):
    """
    Creates a  directory given a path.
    Arguments:
        path: path where the directory will be created.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    return