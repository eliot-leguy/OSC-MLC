from river import base
import numpy as np

class EnsembleMultiLabelClassifier(base.MultiLabelClassifier):
    def __init__(self, models):
        """
        Initialize the ensemble with a list of sub-models.

        Args:
            models: List of tuples, where each tuple is (model_instance, requires_signature).
                    - model_instance: The sub-model object.
                    - requires_signature: Boolean indicating if the model requires a signature in learn_one.
        """
        self.models = models

    def predict_one(self, features):
        """
        Predict probabilities for one sample by averaging sub-model predictions.

        Args:
            features: Dict of feature names to values.

        Returns:
            Dict mapping label names to averaged probability scores.
        """
        # Collect predictions from all sub-models
        all_preds = [model.predict_one(features) for model, _ in self.models]
        print(all_preds)
        # Assume all models predict the same set of labels (keys from the first model's output)
        labels = set().union(*[pred.keys() for pred in all_preds])
        averaged_preds = {}
        for label in labels:
            # Average probabilities for each label across all models
            probs = [preds.get(label, 0.0) for preds in all_preds]
            averaged_preds[label] = np.mean(probs)
        return averaged_preds

    def learn_one(self, features, labels, signature=None):
        """
        Update each sub-model with the current sample.

        Args:
            features: Dict of feature names to values.
            labels: Dict of label names to true values.
            signature: Optional list of label names relevant to the current task (for some models).
        """
        for model, requires_signature in self.models:
            if requires_signature:
                model.learn_one(features, labels, signature)
            else:
                model.learn_one(features, labels)