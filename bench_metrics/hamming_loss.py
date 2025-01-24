__all__ = ["hamming_loss"]


class hamming_loss:
    """Multi-label metrics using the hamming loss

    Calculates the hamming loss for each examples as the ratio of incorrect predicted labels divided by the total number of labels.
    """

    def __init__(self):
        self.sumAccuracy = 0
        self.numberEvaluations = 0

    # Update function. Takes target and predicted label vectors and increments the accuracy sum
    def update(self, target: dict, predicted: dict):
        # Increment the evaluation counter
        self.numberEvaluations += 1

        # Ensure we have all the necessary data in predicted dict
        # to simplify evaluation loop
        targetKeys = target.keys()
        for k in targetKeys:
            if k not in predicted.keys():
                predicted[k] = 0

        # Prepare the counters
        sumIncorrectLabels = 0
        sumLabels = 0

        # Loop on predicted dictionary
        for k in predicted.keys():
            # Get values of the current label for predicted vector
            yPredicted = predicted[k]
            # Get values of the current label for target vector, defaulting to 0 if absent
            yTarget = target.get(k, 0)

            # Increment the incorrect labels counter
            if yTarget != yPredicted:
                sumIncorrectLabels += 1
            # Increment the label counter
            sumLabels += 1

        # Calculate the example's accuracy and increment global accuracy sum
        if sumLabels > 0:
            self.sumAccuracy += float(sumIncorrectLabels) / sumLabels

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        if self.numberEvaluations > 0:
            return self.sumAccuracy / self.numberEvaluations
        else:
            print("Error, no evaluation conducted")
            return 0
