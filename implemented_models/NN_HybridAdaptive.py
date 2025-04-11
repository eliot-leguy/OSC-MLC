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

##################################################
#             NEURAL NETWORK MODEL               #
##################################################
class DeeperNetwork(nn.Module):
    """
    Example deeper network with two hidden layers.
    """
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


##################################################
#               REPLAY MEMORIES                  #
##################################################
class FIFO:
    def __init__(self, size):
        self.size = size
        self.examples_seen = 0
        # We'll store (features_array, labels_array, signature_set)
        self.examples = []

    def add_example(self, x_arr, y_arr, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((x_arr, y_arr, signature))
        else:
            index = self.examples_seen % self.size
            self.examples[index] = (x_arr, y_arr, signature)

    def get_random_example(self):
        if not self.examples:
            return None
        idx = random.randint(0, len(self.examples) - 1)
        return self.examples[idx]


class ReservoirSampling:
    def __init__(self, size):
        self.size = size
        self.examples_seen = 0
        self.examples = []

    def add_example(self, x_arr, y_arr, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((x_arr, y_arr, signature))
        else:
            index = random.randint(0, self.examples_seen)
            if index < self.size:
                self.examples[index] = (x_arr, y_arr, signature)

    def get_random_example(self):
        if not self.examples:
            return None
        idx = random.randint(0, len(self.examples) - 1)
        return self.examples[idx]


##################################################
#         HYBRID MEMORY + ADAPTIVE LOSS          #
##################################################
class NN_HybridAdaptive(base.MultiLabelClassifier):
    """
    Hybrid memory:
      - We keep a FIFO memory AND a reservoir memory
      - We dynamically pick which memory to sample from based on a naive 'policy' about the current label signature.

    Adaptive loss:
      - If the current label signature is small, we do a targeted loss
      - Otherwise, we do a full BCE loss over all labels
    """

    def __init__(
        self,
        feature_size=1006,
        label_size=20,
        learning_rate=0.01,
        hidden1=200,
        hidden2=200,
        size_fifo=1000,
        size_reservoir=1000,
        replay_samples=5,
    ):
        self.feature_size = feature_size
        self.label_size = label_size
        self.learning_rate = learning_rate
        self.replay_samples = replay_samples

        # Setup two memories
        self.fifo_mem = FIFO(size_fifo)
        self.res_mem  = ReservoirSampling(size_reservoir)

        # Build the deeper network
        self.model = DeeperNetwork(feature_size, hidden1, hidden2, label_size).float()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCELoss()

    def learn_one(self, features, labels, signature):
        """
        1) Convert the input into arrays
        2) Decide which memory to sample from
        3) Build a mini-batch from [current sample + replay samples]
        4) Use an 'adaptive loss' approach
        5) Train in one pass
        6) Store the new sample
        """

        # Convert the incoming sample to numeric arrays
        x_arr = numpy.zeros(self.feature_size, dtype=numpy.float32)
        for k, val in features.items():
            idx = int(re.findall(r"\d+", k)[0]) -1 # e.g. "X5" -> 5
            x_arr[idx] = float(val)

        y_arr = numpy.zeros(self.label_size, dtype=numpy.float32)
        for k, val in labels.items():
            if k in signature:
                label_idx = int(re.findall(r"\d+", k)[0])
                y_arr[label_idx] = float(val)

        # We'll build up a mini-batch
        batch_features = [x_arr]
        batch_labels   = [y_arr]
        batch_signs    = [signature]

        # 1) Decide how many labels are in the signature:
        # e.g. if signature is "L0,L1,L3", that's 3 labels out of self.label_size
        sig_size = len(signature)
        # If signature covers big portion of the label space => use FIFO
        # else => use reservoir
        use_fifo = (sig_size >= (0.5 * self.label_size))

        for _ in range(self.replay_samples):
            if use_fifo:
                sample = self.fifo_mem.get_random_example()
            else:
                sample = self.res_mem.get_random_example()
            if sample is None:
                continue
            batch_features.append(sample[0])
            batch_labels.append(sample[1])
            batch_signs.append(sample[2])

        # 2) Convert to Tensors
        features_tensor = torch.tensor(numpy.stack(batch_features), dtype=torch.float32).to(device)
        labels_tensor   = torch.tensor(numpy.stack(batch_labels), dtype=torch.float32).to(device)

        # 3) Build a mask for each sample in the batch if needed
        # We'll do an 'adaptive' approach:
        #    - If signature is "small" => do targeted
        #    - If signature is "large" => do full BCE
        #    - We can do this check *per sample*
        # But for simplicity, let's just decide based on the current sample's signature
        # (so the entire batch uses the same approach).
        do_targeted = (sig_size < 0.25 * self.label_size)

        if do_targeted:
            # We'll build a mask for each row
            masks = []
            for sset in batch_signs:
                row_mask = numpy.zeros(self.label_size, dtype=numpy.float32)
                for i in range(self.label_size):
                    if f"L{i}" in sset:
                        row_mask[i] = 1.0
                masks.append(row_mask)
            mask_tensor = torch.tensor(numpy.stack(masks), dtype=torch.float32, device=device)
        else:
            # Full BCE => mask is all ones
            mask_tensor = torch.ones(labels_tensor.shape, dtype=torch.float32, device=device)

        # 4) Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(features_tensor)  # shape [batch_size, label_size]

        # Apply the mask
        masked_outputs = outputs * mask_tensor

        # Normal BCE
        loss = self.loss_fn(masked_outputs, labels_tensor)
        loss.backward()
        self.optimizer.step()

        # 5) Store the new sample in both memories
        self.fifo_mem.add_example(x_arr, y_arr, signature)
        self.res_mem.add_example(x_arr, y_arr, signature)

    def predict_one(self, features):
        """
        Single-sample inference
        """
        x_arr = numpy.zeros(self.feature_size, dtype=numpy.float32)
        for k, val in features.items():
            idx = int(re.findall(r"\d+", k)[0]) - 1
            x_arr[idx] = float(val)

        x_ten = torch.tensor(x_arr, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(x_ten).cpu().numpy()

        out = {}
        for i, p in enumerate(preds):
            out[f"L{i}"] = p
        return out
