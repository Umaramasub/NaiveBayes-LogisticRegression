import pickle

def read_pickle(filename):
    """Reads a pickle file with name and path of filename, returns the object
    :param filename: Name of the pickle file
    :return: Python object"""

    # opens the file for reading and stores it in a variable file
    file = open(filename, 'r')
    return pickle.load(file)


