import numpy as np
from sklearn.datasets import fetch_openml

from helper.log_helper import get_logger

logger = get_logger(__name__)


def get_mnist_data() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets
    :return: data as x and target as y.
    """
    logger.info("collecting data...")
    mnist = fetch_openml('mnist_784')
    x, y = mnist["data"], mnist["target"]
    logger.info("data collected.")
    return x, y


def get_mnist_data_binary_is5() -> (np.ndarray, np.ndarray):
    """ Get mnist data from openml.org by sklearn.datasets with binary target label
    :return: data as x and target as y.
    """
    x, y = get_mnist_data()
    y = y == '5'
    return x, y


def get_mnist_data_int_binary_is5() -> (np.ndarray, np.ndarray):
    """
    Get mnist data from openml.org by sklearn.datasets with element as int
    :return: data as x and target as y.
    """
    x, y = get_mnist_data_binary_is5()
    x = x.astype(np.int)
    return x, y
