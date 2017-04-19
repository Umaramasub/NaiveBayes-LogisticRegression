import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

def read_pickle(filename):
    """Reads a pickle file with name and path of filename, returns the object
    :param filename: Name of the pickle file
    :return: Python object"""

    # opens the file for reading and stores it in a variable file
    file = open(filename, 'r')
    return pickle.load(file)


def get_files(dir_path):
    """Given an absolute path to a particular directory, return a list of
    absolute paths and filenames for files within the directory.  The intent is
    that the return value of this function can be used with read_pickle()
    without additional munging of paths/filenames.

    :param dir_path: directory path containing files of interest
    :return: list of filenames within directory
    """
    result = []
    # Creates a list with filenames
    list_filenames = os.listdir(os.path.normpath(dir_path))
    # Loop over each filename and concatenate it to the absolute path
    for file_name in list_filenames:
        result.append(os.path.join(os.path.normpath(dir_path), file_name))
    return result

def data(list):
    """Reads a pickle file and returns the array containing the sample data
    :param list: contains a list of all the paths to the sample files
    :return: array containing all the samples for each data file"""
    sample_data = []
    for file in list:
        sample_data.append(read_pickle(file))
    return sample_data

def spilt_data(data, test_size=0.25, random_state=1):
    """Splits the data into training and testing data and returns a tuple containing the train and the test data
    :param numpy array: contains the data with the classifier,default the test size to 0.25 and random_state=1
    :return: tuple with the train data and test data"""
    X_train, X_test, y_train, y_test = train_test_split(data[:,0:2],data[:,2])
    return X_train, X_test, y_train, y_test

def naive_bayes(X_train, X_test, y_train, y_test):
    clf_model= GaussianNB()
    train_model=clf_model.fit(X_train,X_test)
    prediction=clf_model.predict(y_train)

def linear_reg(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
    train_model=regr.fit(X_train, X_test)
    prediction=regr.predict(y_train)

def sensitivity(true_value,predict_value):
    """This function takes three parameters, the true value and the predicted value from the classifier.
    of the test data.The function will return the calculated sensitivity value as a float.
    :param the true value and the predicted value
    :return: calculated sensitivity as float"""

    # Create a confusion matrix to get the rates
    conf_matrix = confusion_matrix(true_value, predict_value)

    # Extract the required values from the matrix
    true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    if true_pos + false_neg == 0:
        result = 0
    else:
        result = float(true_pos) / (true_pos + false_neg)
    return result

def specificity(true_value,predict_value):
    """This function takes three parameters, the true value and the predicted value from the classifier.
    of the test data.The function will return the calculated sensitivity value as a float.
    :param the true value and the predicted value
    :return: calculated specificity as float """

    # Create a confusion matrix to get the rates
    conf_matrix = confusion_matrix(true_value,predict_value)
    true_neg = conf_matrix[0][0]
    false_pos = conf_matrix[0][1]
    if true_neg + false_pos == 0:
        result = 0
    else:
        result = float(true_neg) / (true_neg + false_pos)
    return result








if __name__ == '__main__':
    wd = os.getcwd()
    files=get_files(wd)
    print files
