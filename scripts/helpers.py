import os
def create_dir(path):
    """
    Creates a  directory given a path.
    :param path: path where the directory will be created.
    :return: None
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    return None