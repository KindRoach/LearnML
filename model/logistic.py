import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from helper.data_helper import get_mnist_data
from helper.eval_helper import eval_classification


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def classify(y_hat: float) -> bool:
    return y_hat > 0.5


sigmoid_v = np.vectorize(sigmoid)
classify_v = np.vectorize(classify)


class Logistic(object):
    def __init__(self, alpha=0.01, rounds=5000):
        self.samples = None
        self.labels = None
        self.w = None
        self.alpha = alpha
        self.rounds = rounds

    def fit_gd(self, samples: np.ndarray, labels: np.ndarray):
        self.samples = samples
        self.labels = labels
        self.w = np.ones(self.samples.shape[1])
        for i in range(self.rounds):
            p = sigmoid_v(np.matmul(self.samples, self.w))
            gradient = np.matmul(self.labels - p, self.samples)
            self.w = self.w + self.alpha * (1 - i / self.rounds) * gradient

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return classify_v(np.matmul(samples, self.w))


if __name__ == "__main__":
    x, y = get_mnist_data(n=5000)
    y = y == y[0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    mylo = Logistic()
    mylo.fit_gd(x_train, y_train)
    y_pred = mylo.predict(x_test)
    eval_classification("my", y_test, y_pred)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train)
    y_pred = sgd_clf.predict(x_test)
    eval_classification("sk", y_test, y_pred)
