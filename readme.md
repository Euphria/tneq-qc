# Quantum Circuit Tensor Network (QCTN)

This module provides classes and utilities for representing and simulating quantum circuits as tensor networks using JAX.

## Features

- **QCTN**: Main class for parsing a quantum circuit graph, building the corresponding tensor network, and performing contractions.

## Example Usage

```python
from qctn import QCTN, QCTNHelper

# Generate an example graph
graph = QCTNHelper.generate_example_graph()

# Initialize the QCTN object
qctn = QCTN(graph)

# Print the circuit structure
print(qctn)

# Contract the tensor network (core-only contraction)
result = qctn.contract()
```

## Graph Format

A quantum circuit is described as a multi-line string, where each line represents a qubit and connections to tensor cores (labeled by uppercase letters or CJK characters):

```text
-2-----B-5-C-3-D-----2-
-2-A-4---------D-----2-
-2-A-4-B-7-C-2-D-4-E-2-
-2-A-3-B-6---------E-2-
-2---------C-8-----E-2-
```

- Letters (A, B, C, ...) denote tensor cores.
- Numbers denote the rank (dimension) of each connection.

## API Overview

- `QCTNHelper.generate_example_graph()`: Returns a sample graph string.
- `QCTNHelper.generate_random_example_graph(nqubits, ncores)`: Generates a random graph.
- `QCTN(graph)`: Parses the graph and builds the tensor network.
- `QCTN.contract(attach=None, engine=...)`: Contracts the network, optionally with inputs or another QCTN.

## Requirements

- [JAX](https://github.com/google/jax)
- [NumPy](https://numpy.org/)

## Notes

- The contraction logic relies on an external engine (`ContractorOptEinsum`).
- Some advanced contraction methods are placeholders and need implementation.
