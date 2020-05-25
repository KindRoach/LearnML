import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class HMM:
    def __init__(self, A: np.ndarray, B: np.ndarray, Pi: np.ndarray):
        self.Q = np.arange(A.shape[0])  # n: states
        self.V = np.arange(B.shape[1])  # m: observations

        assert B.shape[0] == self.Q.shape[0]
        assert Pi.shape[0] == self.Q.shape[0]  # shape in 1*n

        self.A = A  # state transition probability matrix, shape in (n,n)
        self.B = B  # observation likelihoods, shape in (n,m)
        self.Pi = Pi  # initial probability distribution, shape in (1,n)

    def generate_sequence(self, d: int, l: int) -> (np.ndarray, np.ndarray):
        Is, Os = [], []
        for _ in range(d):
            state = np.random.choice(self.Q, p=self.Pi)
            I, O = [], []
            for i in range(l):
                obs = np.random.choice(self.V, p=self.B[state])
                O.append(obs)
                I.append(state)
                state = np.random.choice(self.Q, p=self.A[state])
            Is.append(I)
            Os.append(O)
        return np.array(Is), np.array(Os)

    def solve_hidden_state(self, O: np.ndarray) -> np.ndarray:
        N = len(self.Q)
        assert len(O.shape) == 2
        assert N >= O.max() + 1

        D = O.shape[0]
        T = O.shape[1]
        Is = []
        for d in range(D):
            o = O[d]
            delta = np.zeros([T, N])
            psi = np.zeros([T, N], dtype=np.int32)

            # init delta array.
            for i in range(N):
                delta[0, i] = self.Pi[i] * self.B[i, o[0]]

            # DP
            for t in range(1, T):
                for i in range(N):
                    p_j2i = delta[t - 1, :] * self.A[:, i]
                    j = np.argmax(p_j2i)
                    delta[t, i] = p_j2i[j] * self.B[i, o[t]]
                    psi[t, i] = j

            # recover states path I
            I = np.zeros(T, dtype=np.int32)
            I[T - 1] = np.argmax(delta[T - 1, :])
            for t in range(T - 1, 0, -1):
                I[t - 1] = psi[t, I[t]]

            Is.append(I)

        return np.array(Is)

    @staticmethod
    def learn_parameters_from_O_and_I(I: np.ndarray, O: np.ndarray):

        assert len(O.shape) == 2
        assert O.shape == I.shape

        m = O.max() + 1
        n = I.max() + 1

        A = np.zeros([n, n])
        B = np.zeros([n, m])
        Pi = np.zeros([n])

        for i in range(O.shape[0]):
            states = I[i]
            obs = O[i]
            for l in range(states.shape[0] - 1):
                s_now = states[l]
                s_next = states[l + 1]
                A[s_now, s_next] += 1

            for l in range(states.shape[0]):
                s = states[l]
                o = obs[l]
                B[s, o] += 1

            Pi[states[0]] += 1

        A = covert_weight_probability(A)
        B = covert_weight_probability(B)
        Pi = Pi / Pi.sum()

        return HMM(A, B, Pi)


def covert_weight_probability(X: np.ndarray) -> np.ndarray:
    X = X.T / X.sum(axis=1)
    return X.T


def init_parameters_randomly(m: int, n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    A = np.random.random([n, n])
    A = covert_weight_probability(A)

    B = np.random.random([n, m])
    B = covert_weight_probability(B)

    Pi = np.random.random([n])
    Pi = Pi / Pi.sum()

    return A, B, Pi


def init_parameters_const() -> (np.ndarray, np.ndarray, np.ndarray):
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])

    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    Pi = np.array([0.4, 0.4, 0.2])

    return A, B, Pi


def diff_two_model(x: HMM, y: HMM) -> (float, float, float):
    a_error = np.abs(x.A - y.A).mean()
    b_error = np.abs(x.B - y.B).mean()
    pi_error = np.abs(x.Pi - y.Pi).mean()
    return a_error, b_error, pi_error


def diff_two_I(actual: np.ndarray, pred: np.ndarray) -> str:
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)
    return classification_report(actual, pred)


if __name__ == "__main__":
    np.random.seed(42)

    # A, B, Pi = init_parameters_const()
    A, B, Pi = init_parameters_randomly(2, 3)
    model = HMM(A, B, Pi)

    TEST_N = 10000
    LENGTH = 10
    I, O = model.generate_sequence(TEST_N, LENGTH)
    train_I, test_I, train_O, test_O = train_test_split(I, O, test_size=0.3, random_state=42)
    learn_model = HMM.learn_parameters_from_O_and_I(train_I, train_O)

    print("Parameters Error: A = %0.3f, B = %0.3f, Pi = %0.3f" % diff_two_model(learn_model, model))

    I_predict = learn_model.solve_hidden_state(test_O)
    print(diff_two_I(test_I, I_predict))
