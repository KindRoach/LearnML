import math
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from helper.data_helper import get_mnist_data, get_iris_data
from helper.eval_helper import eval_classification
from helper.log_helper import get_logger

logger = get_logger(__name__)


class NaiveBayes(object):

    def __init__(self):
        """
        pfs: probability of every feature show in each class
        pcs: probability of every class show
        """
        self.pf = None
        self.pc = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        logger.info("naive bayes fitting...")
        self.pf = self.__calculate_p(x_train, y_train)
        unique_values, occur_count = np.unique(y_train, return_counts=True)
        self.pc = dict(zip(unique_values, occur_count / len(y_train)))
        logger.info("naive bayes done")

    def predict(self, samples) -> np.ndarray:
        logger.info("naive bayes predicting test set...")
        labels = list()
        for sample in samples:
            labels.append(self.__predict_one(sample))
        logger.info("naive bayes prediction done")
        return np.array(labels)

    def __predict_one(self, sample: np.ndarray):
        max_lnp = float("-inf")
        clazz_predict = None
        for clazz, pci in self.pc.items():
            lnp = math.log(pci)
            for col, val in enumerate(sample):
                pfi = self.pf[clazz][col]
                if val not in pfi:  # pfi[val]==0
                    lnp = float("-inf")
                else:
                    lnp += math.log(pfi[val])

            if lnp >= max_lnp:
                max_lnp = lnp
                clazz_predict = clazz
        return clazz_predict

    def __calculate_p(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate P(x|y) for each class
        result[i][j][v] means the probability of that v is the value of feature_j in class_i
        or mark as P(feature[j]=v|y=i)
        :param x: samples as X
        :param y: labels as y
        :return: Dict from yi to P(x|y=yi)
        """
        unique_y = np.unique(y)
        p_yi = [self.__calculate_p_one_label(x[y == yi]) for yi in unique_y]
        return dict(zip(unique_y, p_yi))

    def __calculate_p_one_label(self, x: np.ndarray) -> List[dict]:
        """
        Calculate P(x) for each feature.
        result[j][v] means the probability of that v is the value of feature_j
        or mark as P(feature[j]=v)
        :param x: x labeled in same class
        :return: List of Dict from xi to P(x=xi)
        """
        ps = list()
        for column in x.T:
            unique_values, occur_count = np.unique(column, return_counts=True)
            probability = occur_count / len(column)
            ps.append(dict(zip(unique_values, probability)))
        return ps


if __name__ == "__main__":
    x, y = get_iris_data()
    x = (10*x).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    mynb = NaiveBayes()
    mynb.fit(x_train, y_train)
    y_pred = mynb.predict(x_test)
    eval_classification("my", y_test, y_pred)

    # GaussianNB from sk will throw error
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_sk = gnb.predict(y_test)
    eval_classification("sk", y_test, y_pred_sk)

