__all__ = ["NN_TLH_sampling_classifier"]

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


######################
#   REPLAY MEMORIES  #
######################
class ReservoirSampling:
    def __init__(self, size):
        self.size = size
        self.examples_seen = 0
        self.examples = []

    def add_example(self, features, labels, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((features, labels, signature))

        else:
            index = random.randint(0, self.examples_seen)
            if index < self.size:
                self.examples[index] = (features, labels, signature)

    def get_random_example(self):
        index = random.randint(0, len(self.examples) - 1)
        return self.examples[index]


#################################
#   NEURAL NETWORK CLASSIFIER   #
#################################
class NN_TLH_sampling_classifier(base.MultiLabelClassifier):
    def __init__(
        self,
        learning_rate=0.01,
        feature_size=1006,
        hidden_sizes=200,
        size_replay_sampling=1000,
        replay_sampling=5,
        label_size=20,
    ):
        self.feature_size = feature_size
        self.label_size = label_size

        # SETTING UP THE RESERVOIR SAMPLING REPLAY MEMORY
        self.sampling_memory = ReservoirSampling(size_replay_sampling)
        self.size_replay_sampling = size_replay_sampling
        self.replay_sampling = replay_sampling

        self.model = NeuralNetwork_WithHiddenLayer(
            feature_size, hidden_sizes, label_size
        )
        self.model = self.model.float()
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # FUNCTION USED TO TRY TO STORE A NEW EXAMPLE INtO THE REPLAY MEMORIES (SAMPLING AND FIFO)
    def store_example(self, features, labels, signature):
        if self.size_replay_sampling > 0:
            self.sampling_memory.add_example(features, labels, signature)

    def learn_one(self, features, labels, signature):
        # Learning current sample
        self.targeted_learn(features, labels, signature)

        # Replay learning
        if (self.size_replay_sampling > 0) and (
            len(self.sampling_memory.examples) > self.replay_sampling
        ):
            for i in range(self.replay_sampling):
                past_example = self.sampling_memory.get_random_example()
                self.targeted_learn(past_example[0], past_example[1], past_example[2])

        self.store_example(features, labels, signature)

    def targeted_learn(self, features, labels, signature):
        # pre-traitement hors tensors GPU et gradient
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(
                re.findall(r"\d+", key)[0]
            )  # "X49" -> "49". # = key pour le fichier test
            new_x[k] = value
        new_x = torch.tensor(new_x, dtype=torch.float32).to(self.device)
        new_y = numpy.zeros(self.label_size)
        for key, value in labels.items():
            if key in signature:
                k = int(re.findall(r"\d+", key)[0])
                new_y[k] = value
        new_y = torch.tensor(new_y, dtype=torch.float32).to(self.device)

        mask = torch.zeros(self.label_size)
        for i in range(self.label_size):
            if "L{}".format(i) in signature:
                mask[i] = 1.0
            else:
                mask[i] = 0

        # operations avec calcul de gradient par pytorch
        self.optimizer.zero_grad()
        outputs = self.model(new_x)
        mask = mask.to(outputs.device)  # Move mask to the same device as outputs (e.g., cuda:0)
        masked_outputs = outputs * mask
        loss = self.loss_fn(masked_outputs, new_y)
        loss.backward()
        self.optimizer.step()

    def predict_one(self, features):
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(re.findall(r"\d+", key)[0])  # "X49" -> "49".
            new_x[k] = value
        new_x = torch.tensor(new_x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(new_x)
        pred_tensor = pred_tensor.cpu().numpy()
        n = 0
        dict_y_pred = dict()
        for j in pred_tensor:
            dict_y_pred["L{}".format(n)] = j
            n += 1
        return dict_y_pred
