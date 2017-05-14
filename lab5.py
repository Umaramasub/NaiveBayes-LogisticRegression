import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing as pp
import numpy


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


def spilt_data(data, test_size=0.33, random_state=1):
    """Splits the data into training and testing data and returns a tuple containing the train and the test data
    :param numpy array: contains the data with the classifier,default the test size to 0.33 and random_state=1
    :return: tuple with the train data and test data"""
    # Scale the data as weight and A1C scores have high numeric difference
    scaler_object = pp.StandardScaler()
    scaled_data = scaler_object.fit_transform(data[:, 0:2])

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(scaled_data[:, 0:2], data[:, 2])
    return X_train, X_test, y_train, y_test


def naive_bayes(X_train, X_test, y_train, y_test):
    """Given the training and test data, it trains the data using Naive Bayes model, the model is then tested using the testing data.Based
    on the results of the testing, the performance metrics of the model is calculated.
     The function will return a dictionary containing all the performance metrics to be used for plotting ROC curve
    :param Tuple containing the test and the training data
    :return: a dictionary containing model, sensitivity,specificity,Accuracy,F1-score,Auc, the true values of test data, y score"""

    # Create a Gaussian NaiveBayes model instance
    clf_model = GaussianNB()

    # Fit the model to the training data
    train_model = clf_model.fit(X_train, y_train)

    # Check the performance of the model using the test data
    y_pred = train_model.predict(X_test)

    # Calculate the F1 score
    f1score = f1_score(y_test, y_pred, average='macro')

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the y score of the positive class
    y_score = train_model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_score[:, 1])

    # Create a confusion matric to calculate sensitivity and specificity
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_neg = conf_matrix[0][0]
    false_pos = conf_matrix[0][1]
    if true_neg + false_pos == 0:
        spec = 0
    else:
        spec = float(true_neg) / (true_neg + false_pos)
        true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    if true_pos + false_neg == 0:
        sens = 0
    else:
        sens = float(true_pos) / (true_pos + false_neg)

    # Create a dictionary to return the calculated performance matrix
    NB_dict = {
        'Model': 'Naive Bayes',
        'Sensitivity': sens,
        'Specificity': spec,
        'Accuracy': accuracy,
        'F1 Score': f1score,
        'Auc': auc,
        'Y_test': y_test,
        'Y_score': y_score[:, 1]
    }
    return NB_dict


def log_reg(X_train, X_test, y_train, y_test):
    """Given the training and test data, it trains the data using Logistic Regression model, the model is then tested using testing data .Based
    on the results of the testing, the performance metrics of the model is calculated.
     The function will return a dictionary containing all the performance metrics to be used for plotting ROC curve
    :param Tuple containing the test and the training data
    :return: a dictionary containing model, sensitivity,specificity,Accuracy,F1-score,Auc, the true values of test data, y score"""

    # Create a Logistic Regression  model instance
    regr = LogisticRegression()

    # Fit the model to the training data
    train_model = regr.fit(X_train, y_train)

    # Check the performance of the model using the test data
    y_pred = regr.predict(X_test)

    # Calculate the F1 score
    f1score = f1_score(y_test, y_pred, average='macro')

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the Y score of the model
    y_score = train_model.decision_function(X_test)

    # Calculate the auc score
    auc = roc_auc_score(y_test, y_score)

    # Create an confusion matrix to calculate sensitivity and specificity
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_neg = conf_matrix[0][0]
    false_pos = conf_matrix[0][1]
    if true_neg + false_pos == 0:
        spec = 0
    else:
        spec = float(true_neg) / (true_neg + false_pos)
        true_pos = conf_matrix[1][1]
    false_neg = conf_matrix[1][0]
    if true_pos + false_neg == 0:
        sens = 0
    else:
        sens = float(true_pos) / (true_pos + false_neg)

    # Create a dictionary with all the performance metrics
    LR_dict = {
        'Model': 'LogisticRegression',
        'Sensitivity': sens,
        'Specificity': spec,
        'Accuracy': accuracy,
        'F1 Score': f1score,
        'Auc': auc,
        'Y_test': y_test,
        'Y_score': y_score
    }
    return LR_dict


def plot_roc(NB_dict, LR_dict):
    """Given the dictionary with the performance metrics and the model, it plots the ROC curve for each of the model
     The function will show subplots comparing the performance of each of the model on the same data
    :param Dictionaries containing the parameters of each of the model
    :return: Subplots of the ROC curve for each of the model"""

    # Create an instance of plot
    fig = plt.figure()

    # Create the axis for the first subplot
    ax = fig.add_subplot(1, 2, 1)

    # Get the required metrics for plotting ROC curve for Naive Bayes model
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(NB_dict['Y_test'], NB_dict['Y_score'])
    ax.plot(false_positive_rate, true_positive_rate, 'b-',
            label='Sensitivity:%.2f\nSpecificity:%.2f\nAccuracy:%.2f\nF1 score:%.2f\nAuc:%.2f' % (
                NB_dict['Sensitivity'], NB_dict['Specificity'], NB_dict['Accuracy'], NB_dict['F1 Score'],
                NB_dict['Auc']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4, fontsize='xx-large')
    plt.title('Naive Bayes Model')

    # Create the axis for the second subplot
    ax = fig.add_subplot(1, 2, 2)

    # Get the required metrics for plotting ROC curve for Logistic Regression model
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(LR_dict['Y_test'], LR_dict['Y_score'])
    ax.plot(false_positive_rate, true_positive_rate, 'b-',
            label='Senstivity:%.2f\nSpecificity:%.2f\nAccuracy:%.2f\nF1 score:%.2f\nAuc:%.2f' % (
                LR_dict['Sensitivity'], LR_dict['Specificity'], LR_dict['Accuracy'], LR_dict['F1 Score'],
                LR_dict['Auc']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression Model')
    plt.legend(loc=4, fontsize='xx-large')
    plt.show()


if __name__ == '__main__':
    wd = os.getcwd()
    files = get_files(wd)
    data_files = files[1:6]
    for file in data_files:
        data = read_pickle(file)
        X_train, X_test, y_train, y_test = spilt_data(data)
        dict1 = naive_bayes(X_train, X_test, y_train, y_test)
        dict2 = log_reg(X_train, X_test, y_train, y_test)
        plot_roc(dict1, dict2)
