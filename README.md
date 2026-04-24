# Self-Pruning Neural Network

## Overview
This project implements a self-pruning neural network using PyTorch. The model learns to remove unnecessary weights during training using learnable gates.

## Key Concepts

- **Gates**: Each weight has a learnable gate value (0 to 1 using sigmoid)
- **Pruning**: Weights with low gate values are effectively removed
- **Sparsity**: Encouraged using L1 regularization on gate values

## Approach

- Custom `PrunableLinear` layer created
- Each weight multiplied by a sigmoid gate
- Loss function:
  
  Total Loss = Classification Loss + λ * Sparsity Loss

- Sparsity loss = sum of all gate values (L1 norm)

## Current Status

This is a partial implementation focusing on:
- Model architecture
- Gated weight mechanism

Future work:
- Training on CIFAR-10
- Sparsity evaluation
- Accuracy vs sparsity comparison

## Tech Stack
- Python
- PyTorch
