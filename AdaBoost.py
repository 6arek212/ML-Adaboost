import numpy as np
import math


def load_data(filename='./squares.txt'):
    data = []
    labels = []
    with open(filename, 'r') as f:
        lines_array = f.readlines()
        for line in lines_array:
            line_as_arr = line.split(" ")
            labels.append(line_as_arr[-1].split('\n')[0])
            data.append(line_as_arr[:-1])

        labels_arr = np.array(labels, dtype=float)
        mask = (labels_arr == 0)
        labels_arr[mask] = -1
    return np.array(data, dtype=float), labels_arr


def split_train_test(X, y, seed, tests_ratio=0.5):
    random_state = np.random.RandomState(seed)
    shuffled_indices = random_state.permutation(len(X))
    test_set_size = int(len(X) * tests_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return (X[train_indices], y[train_indices]), (X[test_indices], y[test_indices])


class AdaBoost:

    def __init__(self, use_line_rules=True, iterations=50, k_rules=8):
        self._empirical_errors = np.zeros(shape=k_rules)
        self._actual_errors = np.zeros(shape=k_rules)
        self._iterations = iterations
        self._k_rules = k_rules
        self._use_line_rules = use_line_rules
        self._rules = []
        self._rule_weights = []
        self._point_weights = []
        self._selected_rules = []

    def _create_line_hypothesis_set(self, X):
        self._rules = []

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                point1, point2 = X[i], X[j]
                (x1, y1), (x2, y2) = point1, point2
                if x2 - x1 == 0:
                    break
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
                self._rules.append((a, b))

        self._rule_weights = np.zeros(shape=len(self._rules))

    def _create_circle_hypothesis_set(self, X):
        self._rules = []

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                point1, point2 = X[i], X[j]
                (x1, y1), (x2, y2) = point1, point2

                radius = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                self._rules.append((radius, (x1, y1)))

        self._rule_weights = np.zeros(shape=len(self._rules))

    def run(self, X, y):
        for i in range(self._iterations):
            # 1. Initialize point weights (uniformly)
            (train_X, train_y), (test_X, test_y) = split_train_test(
                X, y, seed=i + 1)
            n_samples, n_features = train_X.shape
            self._point_weights = np.ones(shape=n_samples) * (1 / n_samples)

            # create hypotheses set + init rule weights (alphas)
            if self._use_line_rules:
                self._create_line_hypothesis_set(train_X)
            else:
                self._create_circle_hypothesis_set(train_X)

            self._selected_rules = []

            for t in range(self._k_rules):  # 2. Iterate until k (rules)
                # 3. Compute weighted error for each h
                weighted_errors = []
                for h in self._rules:
                    predictions = self._predict(train_X, *h)
                    indicators = np.array(
                        [1 if y_pred != y_real else 0 for (y_pred, y_real) in zip(predictions, train_y)])
                    error = np.dot(self._point_weights, indicators)
                    weighted_errors.append(error)
                # 4. Select classifier with min weighted error
                classifier_index = np.argmin(weighted_errors)
                classifier = self._rules[classifier_index]
                classifier_error = weighted_errors[classifier_index]

                # 5. Set classifier weight (alpha_t) based on its error
                alpha_t = 0.5 * \
                    math.log((1 - classifier_error) / classifier_error)
                self._rule_weights[classifier_index] = alpha_t

                # 6. Update point weights
                predictions = self._predict(train_X, *classifier)
                new_point_weights = self._point_weights * \
                    np.exp(-alpha_t * predictions * train_y)
                normalizing_constant = 1 / sum(new_point_weights)
                new_point_weights *= normalizing_constant
                self._point_weights = new_point_weights

                # add rule and alpha to selected rules
                self._selected_rules.append((alpha_t, classifier))

                # compute empirical and actual error
                self._compute_error(train_X, train_y, test_X, test_y, curr_k=t)

        return self

    def _compute_error(self, train_X, train_y, test_X, test_y, curr_k):
        train_accumulator = np.zeros(shape=len(train_X))
        test_accumulator = np.zeros(shape=len(test_X))

        for alpha, classifier in self._selected_rules:
            train_accumulator += alpha * self._predict(train_X, *classifier)
            test_accumulator += alpha * self._predict(test_X, *classifier)

        # update empiracle H_k error
        H_k_empirical = np.sign(train_accumulator)
        train_indicators = [H_k_empirical != train_y]
        empirical_err = (np.sum(train_indicators) / self._iterations) / len(train_y)  # moving average
        self._empirical_errors[curr_k] += empirical_err

        # update actual H_k error
        H_k_actual = np.sign(test_accumulator)
        test_indicators = [H_k_actual != test_y]
        actual_err = (np.sum(test_indicators) / self._iterations) / len(test_y)  # moving average
        self._actual_errors[curr_k] += actual_err

    def _errors(self):
        empirical_err = np.round(self._empirical_errors, 2)
        actual_err = np.round(self._actual_errors, 2)
        return empirical_err, actual_err

    def _predict(self, X, *data):
        if (self._use_line_rules):
            return self._predict_line(X, *data)
        return self._predict_circle(X, *data)

    def _predict_line(self, X, m, b):
        return np.where(m * X[:, 0] + b >= X[:, 1], 1, -1)

    def _predict_circle(self, X, radius, center_point):
        return np.where(np.linalg.norm(center_point - X, axis=1) < radius, 1, -1)


if __name__ == '__main__':
    print()

    use_line_rules = True
    print(f'Running Adaboost using {"Lines" if use_line_rules else "Circles" }')

    X, y = load_data()
    adaboost = AdaBoost(use_line_rules=use_line_rules)
    adaboost.run(X, y)
    empirical_errors, actual_errors = adaboost._errors()

    print(f"Empirical Errors: {empirical_errors}")
    print(f"Actual Errors: {actual_errors}")

    diff = np.array(actual_errors) - np.array(empirical_errors)
    print(f"Difference between Actual and Empirical Errors:\n{diff}")

    print()