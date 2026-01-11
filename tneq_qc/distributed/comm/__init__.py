"""
Communication module for distributed training.

Provides communication primitives for tensor synchronization:
- CommBase: Abstract interface for all communication backends
- CommMPI: MPI-based communication (numpy arrays)
- CommTorch: PyTorch distributed communication (torch tensors)
- Factory for creating different backends
"""

# Interface and common types
from .comm_interface import (
    CommBase,
    ReduceOp,
    DistributedContext,
    AsyncHandle,
)

# MPI backend (numpy arrays)
from .comm_mpi import (
    CommMPI,
    MockCommMPI,
    MPIAsyncHandle,
    get_comm_mpi,
    # Backward compatibility aliases
    MPIBackend,
    MockMPIBackend,
    get_mpi_backend,
)

# PyTorch backend (torch tensors)
from .comm_torch import (
    CommTorch,
    MockCommTorch,
    TorchAsyncHandle,
    get_comm_torch,
    # Backward compatibility aliases
    TorchBackend,
    MockTorchBackend,
    get_torch_backend,
)

# Factory
from .comm_factory import (
    get_comm_backend,
    get_auto_backend,
    create_comm_mpi,
    create_comm_torch,
    get_mock_backend,
    detect_best_backend,
    CommBackendType,
    # Backward compatibility aliases
    get_mpi_backend as create_mpi_backend,
    get_torch_backend as create_torch_backend,
)

# Backward compatibility aliases
CommBackendBase = CommBase
get_backend = get_comm_mpi

__all__ = [
    # Interface and common types
    'CommBase',
    'ReduceOp',
    'DistributedContext',
    'AsyncHandle',
    
    # MPI backend
    'CommMPI',
    'MockCommMPI',
    'MPIAsyncHandle',
    'get_comm_mpi',
    
    # PyTorch backend
    'CommTorch',
    'MockCommTorch',
    'TorchAsyncHandle',
    'get_comm_torch',
    
    # Factory
    'get_comm_backend',
    'get_auto_backend',
    'create_comm_mpi',
    'create_comm_torch',
    'get_mock_backend',
    'detect_best_backend',
    'CommBackendType',
    
    # Backward compatibility aliases
    'CommBackendBase',
    'MPIBackend',
    'MockMPIBackend',
    'get_mpi_backend',
    'TorchBackend',
    'MockTorchBackend',
    'get_torch_backend',
    'create_mpi_backend',
    'create_torch_backend',
    'get_backend',
]
