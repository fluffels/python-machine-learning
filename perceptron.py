import numpy


class Perceptron:
    errors: list
    learning_rate: float
    max_iterations: int
    weights: numpy.ndarray

    def __init__(self, learning_rate=0.01, max_iterations=10):
        self.learning_rate = learning_rate
        self.max_iterations = 10

    def train(self, samples: numpy.ndarray, targets: numpy.ndarray):
        self.weights = numpy.zeros(samples.shape[1] + 1)
        self.errors = []

        for _ in range(self.max_iterations):
            error = 0
            for features, target in zip(samples, targets):
                delta = target - self.predict(features)
                update = delta * self.learning_rate
                self.weights[0] += update
                self.weights[1:] += update * features
                if update != 0:
                    error += 1
            self.errors.append(error)

    def output(self, inputs: numpy.ndarray):
        result = self.weights[0] + numpy.dot(inputs, self.weights[1:])
        return result

    def predict(self, inputs: numpy.ndarray):
        output = self.output(inputs)
        result = numpy.where(output >= 0, 1, -1)
        return result
