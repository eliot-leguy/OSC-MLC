__all__ = ["f_measure"]


class f_measure:
    """Multi-label metrics using the F-mesure.

    Calculates the F-mesure for each examples.
    """

    def __init__(self):
        self.sumFmeasure = 0
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
        sumIntersection = 0
        sumPredLabels = 0
        sumTrueLabels = 0

        # Loop on predicted dictionary
        for k in predicted.keys():
            # Get values of the current label for predicted vector
            yPredicted = predicted[k]
            # Get values of the current label for target vector, defaulting to 0 if absent
            yTarget = target.get(k, 0)

            # Increment the incorrect labels counter
            if yTarget == True and yPredicted == True:
                sumIntersection += 1

            # Increment the label counter
            if yTarget == True:
                sumTrueLabels += 1

            if yPredicted == True:
                sumPredLabels += 1

        # Calculate the example's accuracy and increment global accuracy sum
        if (sumTrueLabels + sumPredLabels) > 0:
            self.sumFmeasure += float(sumIntersection) / (sumTrueLabels + sumPredLabels)

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        if self.numberEvaluations > 0:
            return self.sumFmeasure / self.numberEvaluations
        else:
            print("Error, no evaluation conducted")
            return 0
