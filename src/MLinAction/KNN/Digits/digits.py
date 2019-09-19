import math
import os


class DigitsData:
    def __init__(self, digit, vector, file_path):
        self.digit = digit
        self.vector = vector
        self.file_path = file_path


class DigitModel:
    def __init__(self, train_data: list, step_n: int):
        self.train_data = train_data
        self.step_n = min(step_n, 100)

    def guess_digit(self, digit: DigitsData):
        neighbours = {}
        for x in self.train_data:
            neighbours[x] = get_distance(digit.vector, x.vector)
        neighbours = sorted(neighbours.items(), key=lambda kv: kv[1])
        neighbours = [x[0].digit for x in neighbours[0:self.step_n]]
        return max(neighbours, key=neighbours.count)


def file2vector(file_path: str) -> list:
    """
    :rtype: list of int
    :param file_path: the file path to one digit data.
    :return: A vector representing digit.
    """
    vector = []
    with open(file_path) as f:
        for line in f:
            vector.extend([int(x) for x in line.replace('\n', '')])
    return vector


def get_data_set(data_path: str) -> list:
    """
    :rtype: list of DigitsData
    :param data_path: path to the data dir
    :return: All DigitsData under given path
    """
    data_set = []
    print(data_path)
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        vector = file2vector(file_path)
        digit = int(file_name.split('_')[0])
        data_set.append(DigitsData(digit, vector, file_path))
    return data_set


def get_distance(vector1: list, vector2: list) -> float:
    sqr = 0
    for i in range(len(vector1)):
        sqr += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(sqr)


train_set = get_data_set(os.path.join("data", "trainingDigits"))
test_set = get_data_set(os.path.join("data", "testDigits"))
model = DigitModel(train_set, 10)
right = 0
wrong = 0
wrong_list = []
for x in test_set:
    guess = model.guess_digit(x)
    if guess == x.digit:
        right += 1
    else:
        wrong += 1
        wrong_list.append((x, guess))
    print("Working:%d/%d" % (right + wrong, len(test_set)))
print("-------------------------Report-------------------------")
print("Right:%f, Wrong:%f" % (right / len(test_set), wrong / len(test_set)))
for x in wrong_list:
    print("Guess=%d, Actual=%d, File Path: %s" % (x[1], x[0].digit, x[0].file_path))
