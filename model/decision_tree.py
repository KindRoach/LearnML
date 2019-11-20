import math
from typing import List

import numpy as np
from scipy import stats

from helper.log_helper import get_logger

logger = get_logger(__name__)


def calculate_entropy(samples: np.ndarray) -> float:
    # unique_values, occur_count = np.unique(group, return_counts=True)
    # total_entropy = 0.
    # total_count = group.shape[0]
    # for count in occur_count:
    #     probability = count / total_count
    #     entropy = - math.log(probability, 2)
    #     total_entropy += entropy * probability
    # return total_entropy

    # A elegant way to calculate entropy
    labels = samples[:, -1]
    unique_values, occur_count = np.unique(labels, return_counts=True)
    probability = occur_count / len(labels)
    return stats.entropy(probability, base=2)


def split_sets(samples: np.ndarray, feature_index: int) -> List[np.ndarray]:
    """ Split one sample set to several sub sets by given feature.
    :argument samples: [X|Y] as numpy array
    :argument feature_index: The index of feature used to split sample set
    :return: sets by feature
    """
    features = samples[:, feature_index]
    unique_values = np.unique(features)
    sets_by_feature = list()
    for value in unique_values:
        # All samples where row[i] == value
        set_i = samples[samples[:, feature_index] == value]
        sets_by_feature.append(set_i)
    return sets_by_feature


if __name__ == "__main__":
    test_group = np.array([[1, 1, 1],
                           [0, 1, 1],
                           [1, 0, 0]])
    logger.info(calculate_entropy(test_group))
    logger.info(split_sets(test_group, 0))
