import numpy as np
from numpy import trapz

__all__ = ["learning_speed"]


class learning_speed:

    def __init__(self):
        self.accuracies = np.array([])
        self.learning_speed = 0

    # Update function. Takes target and predicted label vectors and increments the accuracy sum
    def update(self, new_accuracy: float):
        self.accuracies = np.append(self.accuracies, new_accuracy)
        self.learning_speed = trapz(self.accuracies, dx=1)

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        return self.learning_speed
