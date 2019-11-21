import numpy as np

from sklearn.metrics import classification_report
from helper.log_helper import get_logger

logger = get_logger(__name__)


def eval_classification(name: str, actual: np.ndarray, pred: np.ndarray):
    logger.info(f"{name} result:\n{classification_report(actual, pred)}")
