import numpy
import torch
from torch import nn
import re
import random
from river import base

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

############################
#   NEURAL NETWORK MODEL   #
############################
class NeuralNetwork_WithHiddenLayer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


#############################
#   ATTENTION-BASED MEMORY  #
#############################
class ReservoirAttention:
    """
    Stores examples in typical reservoir fashion,
    but offers get_attention_examples() to pick
    top-K closest matches to the current sample.
    """
    def __init__(self, size, feature_size):
        self.size = size
        self.feature_size = feature_size
        self.examples_seen = 0
        # We'll store (x_array, y_array, signature)
        self.examples = []

    def add_example(self, x_array, y_array, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((x_array, y_array, signature))
        else:
            index = random.randint(0, self.examples_seen)
            if index < self.size:
                self.examples[index] = (x_array, y_array, signature)

    def get_attention_examples(self, current_x_array, top_k=5):
        if not self.examples:
            return []
        # Distances to current_x_array
        dists = []
        for i, (old_x, old_y, sig) in enumerate(self.examples):
            dist_val = numpy.sum((old_x - current_x_array)**2)
            dists.append((dist_val, i))
        dists.sort(key=lambda x: x[0])  # ascending order
        # Return top_k
        selected = [self.examples[idx] for _, idx in dists[:top_k]]
        return selected


#########################################
#   ATTENTION-BASED NEURAL NETWORK CLF  #
#########################################
class NN_TLH_attention(base.MultiLabelClassifier):
    def __init__(
        self,
        learning_rate=0.01,
        feature_size=1006,
        hidden_sizes=200,
        size_replay=1000,    # reservoir size
        replay_k=5,          # how many neighbors to replay each time
        label_size=20,
    ):
        self.feature_size = feature_size
        self.label_size = label_size
        self.feature_mapping = None  # Initialize feature mapping

        # Set up an attention-based reservoir
        self.memory = ReservoirAttention(size=size_replay, feature_size=feature_size)
        self.size_replay = size_replay
        self.replay_k = replay_k

        self.model = NeuralNetwork_WithHiddenLayer(feature_size, hidden_sizes, label_size)
        self.model = self.model.float()
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = device
        self.model.to(self.device)

    def predict_one(self, features):
        # Set feature mapping on first call
        if self.feature_mapping is None:
            self.feature_mapping = {key: i for i, key in enumerate(features.keys())}
            # Verify that the number of features matches feature_size
            if len(self.feature_mapping) != self.feature_size:
                raise ValueError(
                    f"Number of features ({len(self.feature_mapping)}) does not match "
                    f"feature_size ({self.feature_size})"
                )

        x_array = numpy.zeros(self.feature_size, dtype=numpy.float32)
        for key, value in features.items():
            idx = self.feature_mapping[key]
            x_array[idx] = float(value)

        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(x_tensor).cpu().numpy()

        out = {f"L{i}": p for i, p in enumerate(preds)}
        return out

    def _to_arrays(self, features, labels, signature):
        # Ensure feature_mapping is set (should be by predict_one, but for safety)
        if self.feature_mapping is None:
            self.feature_mapping = {key: i for i, key in enumerate(features.keys())}
            if len(self.feature_mapping) != self.feature_size:
                raise ValueError(
                    f"Number of features ({len(self.feature_mapping)}) does not match "
                    f"feature_size ({self.feature_size})"
                )

        x_arr = numpy.zeros(self.feature_size, dtype=numpy.float32)
        for key, val in features.items():
            idx = self.feature_mapping[key]
            x_arr[idx] = float(val)

        y_arr = numpy.zeros(self.label_size, dtype=numpy.float32)
        for key, val in labels.items():
            if key in signature:
                idx = int(re.findall(r"\d+", key)[0])
                y_arr[idx] = float(val)

        return x_arr, y_arr

    def learn_one(self, features, labels, signature):
        x_array, y_array = self._to_arrays(features, labels, signature)
        self._targeted_learn(x_array, y_array, signature)
        if len(self.memory.examples) >= self.replay_k:
            neighbors = self.memory.get_attention_examples(x_array, top_k=self.replay_k)
            for (old_x, old_y, old_sig) in neighbors:
                self._targeted_learn(old_x, old_y, old_sig)
        self.memory.add_example(x_array, y_array, signature)

    def _targeted_learn(self, x_array, y_array, signature):
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_array, dtype=torch.float32).to(self.device)
        mask = torch.zeros(self.label_size, dtype=torch.float32)
        for i in range(self.label_size):
            if f"L{i}" in signature:
                mask[i] = 1.0
        mask = mask.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x_tensor)
        masked_outputs = outputs * mask
        loss = self.loss_fn(masked_outputs, y_tensor)
        loss.backward()
        self.optimizer.step()