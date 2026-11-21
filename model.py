"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), 
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )

# Step 2 - build_synthetic_dataset
def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    # TODO: build a seeded synthetic dataset of (features, labels) tensors
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, input_size, generator=gen, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (num_samples,), generator=gen, dtype=torch.int64)

    return features, labels

# Step 3 - train_test_split_dataset
def train_test_split_dataset(features, labels, test_fraction, seed):
    # TODO: seeded shuffle of row indices, then slice into train and test sets
    num_rows = features.shape[0]
    num_test = int(num_rows * test_fraction)
    num_train = num_rows - num_test

    #seeded generatr for shuffling
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)

    shuffled_indices = torch.randperm(num_rows, generator=generator, device=features.device)

    #slice into paritions
    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:] 

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
def partition_data_iid(train_features, train_labels, num_clients, seed):
  num_rows = train_features.shape[0]
  generator = torch.Generator(device=train_features.device)
  generator.manual_seed(seed)

  shuffled_indices = torch.randperm(
      num_rows, generator=generator, device=train_features.device
  )

  k = max(1, num_clients)
  client_features = torch.tensor_split(
      train_features[shuffled_indices], k, dim=0
  )
  client_labels = torch.tensor_split(train_labels[shuffled_indices], k, dim=0)

  return list(zip(client_features, client_labels))

# Step 5 - partition_data_non_iid
import torch


def partition_data_non_iid(
    train_features, train_labels, num_clients, shards_per_client, seed
):
  # 1. Sort all row indices by label
  sorted_indices = torch.argsort(train_labels, stable=True)

  # 2. Split sorted indices into contiguous label-grouped shards
  total_shards = num_clients * shards_per_client
  shard_indices = torch.tensor_split(sorted_indices, total_shards, dim=0)

  # 3. Shuffle the shard IDs (0 .. total_shards - 1) once
  generator = torch.Generator(device=train_features.device)
  generator.manual_seed(seed)
  shuffled_shard_ids = torch.randperm(
      total_shards, generator=generator, device=train_features.device
  ).tolist()

  # 4. Deal shards_per_client shards to each client and concatenate
  partitions = []
  for c in range(num_clients):
    start = c * shards_per_client
    end = start + shards_per_client
    client_shard_ids = shuffled_shard_ids[start:end]

    client_idx = torch.cat([shard_indices[i] for i in client_shard_ids], dim=0)
    partitions.append((train_features[client_idx], train_labels[client_idx]))

  return partitions

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    # TODO: return a list of per-client sample counts in the same order
    return [int(features.shape[0]) for features, _ in client_partitions]

# Step 7 - iterate_client_batches (not yet solved)
# TODO: implement

# Step 8 - compute_batch_loss (not yet solved)
# TODO: implement

# Step 9 - local_sgd_step (not yet solved)
# TODO: implement

# Step 10 - train_client_local (not yet solved)
# TODO: implement

# Step 11 - clone_model_state (not yet solved)
# TODO: implement

# Step 12 - load_model_state (not yet solved)
# TODO: implement

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

