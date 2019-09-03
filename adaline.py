import numpy


class AdalineGradientDescent:
    costs: list
    weights: numpy.ndarray

    def train(self, examples: numpy.ndarray, targets: numpy.ndarray,
              training_speed: float = 0.01, training_iterations: int = 50):
        self.costs = []
        self.weights = numpy.zeros(1 + examples.shape[1])

        for _ in range(training_iterations):
            outputs = []
            for example in examples:
                output = self.output(example)
                outputs.append(output)
            errors = targets - outputs
            self.weights[0] += training_speed * errors.sum()
            self.weights[1:] += training_speed * examples.T.dot(errors)
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)

    def output(self, features):
        result = self.weights[0] + numpy.dot(features, self.weights[1:])
        return result

    def predict(self, features):
        result = numpy.where(self.output(features) >= 0.0, 1, -1)
        return result


class AdalineStochasticGradientDescent(AdalineGradientDescent):
    def train(self, examples: numpy.ndarray, targets: numpy.ndarray,
              training_speed: float = 0.01, training_iterations: int = 50):
        self.costs = []
        self.weights = numpy.zeros(1 + examples.shape[1])
        for _ in range(training_iterations):
            costs = []
            for example, target in zip(examples, targets):
                output = self.output(example)
                error = target - output
                costs.append(error**2 / 2.0)
                self.weights[0] += training_speed * error
                self.weights[1:] += example * training_speed * error
            self.costs.append(sum(costs) / len(costs))
