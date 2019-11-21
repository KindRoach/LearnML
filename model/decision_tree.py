import sys

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from helper.data_helper import get_mnist_data, get_iris_data
from helper.eval_helper import eval_classification
from helper.log_helper import get_logger

logger = get_logger(__name__)
sys.setrecursionlimit(2000)


class TreeNode(object):
    def __init__(self, label=None, children: dict = None, feature: int = None):
        self.label = label
        self.children = children
        self.feature = feature

    @property
    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTree(object):
    def __init__(self):
        self.root = None

    def fit(self, samples: np.ndarray, labels: np.ndarray):
        logger.info("decision tree fitting...")
        self.root = self.__fit_tree(samples, labels)
        logger.info("decision tree done")

    def __fit_tree(self, samples: np.ndarray, labels: np.ndarray) -> TreeNode:
        """
        Construct a decision tree by given train data
        :param samples: X
        :param labels: Y
        :return: root node of tree
        """
        if len(np.unique(labels)) == 1:
            return TreeNode(label=labels[0])
        feature = self.__find_best_feature(samples, labels)
        split_sets = self.__split_sets(samples, labels, feature)
        children = dict()
        for samples_i, labels_i in split_sets:
            key = samples_i[0][feature]
            children[key] = self.__fit_tree(samples_i, labels_i)
        return TreeNode(children=children, feature=feature)

    def __find_best_feature(self, samples: np.ndarray, labels: np.ndarray) -> int:
        """
        Find the feature that minimum sum entropy of sub sets
        :param samples: X
        :param labels: Y
        :return: index of best feature
        """
        min_entropy = float("inf")
        best_feature = .1
        for i in range(samples.shape[1]):
            spilt_sets = self.__split_sets(samples, labels, i)
            sub_entropy = [self.__calculate_entropy(item[1]) for item in spilt_sets]
            sum_entropy = sum(sub_entropy)
            if min_entropy > sum_entropy and len(spilt_sets) > 1:
                min_entropy = sum_entropy
                best_feature = i
        return best_feature

    def __calculate_entropy(self, labels: np.ndarray) -> float:
        # A elegant way to calculate entropy
        unique_values, occur_count = np.unique(labels, return_counts=True)
        probability = occur_count / len(labels)
        return stats.entropy(probability, base=2)

    def __split_sets(self, samples: np.ndarray, labels: np.ndarray, feature_index: int) -> list:
        """
        Split one sample set to several sub sets by given feature
        :param samples: X
        :param labels: Y
        :param feature_index: index of feature used to split sample set
        :return: sets by feature as list of (samples_i, labels_i)
        """
        features = samples[:, feature_index]
        unique_values = np.unique(features)
        sets_by_feature = list()
        for value in unique_values:
            # find row[i] == value
            bool_index = samples[:, feature_index] == value
            samples_i = samples[bool_index]
            labels_i = labels[bool_index]
            sets_by_feature.append((samples_i, labels_i))
        return sets_by_feature

    def predict(self, samples: np.ndarray):
        """
        Predict samples by given decision tree
        :param samples: X
        :return: pectic labels.
        """
        logger.info("decision tree predicting test set...")
        labels = list()
        for sample in samples:
            labels.append(self.__predict_one(sample, self.root))
        logger.info("decision tree prediction done")
        return np.array(labels)

    def __predict_one(self, sample: np.ndarray, tree: TreeNode):
        """
        Predict one sample by given decision tree
        :param sample: x
        :param tree: root node of decision tree.
        :return: pectic label.
        """
        if tree.is_leaf:
            return tree.label
        # find key closest to sample[tree.feature]
        key = min(tree.children.keys(), key=lambda x: abs(x - sample[tree.feature]))
        return self.__predict_one(sample, tree.children[key])


if __name__ == "__main__":
    x, y = get_iris_data()
    x = (10 * x).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    my_tree = DecisionTree()
    my_tree.fit(x_train, y_train)
    y_pred = my_tree.predict(x_test)
    eval_classification("my", y_test, y_pred)

    sk_tree = DecisionTreeClassifier(random_state=42)
    sk_tree.fit(x_train, y_train)
    y_pred_sk = sk_tree.predict(x_test)
    eval_classification("sk", y_test, y_pred_sk)
