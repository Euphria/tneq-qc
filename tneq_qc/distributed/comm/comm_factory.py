"""
Communication Backend Factory

Provides factory functions to create communication backends for distributed training.
Supports different backend types:
- "mpi": MPI-based communication using mpi4py (numpy arrays)
- "torch": PyTorch distributed communication (torch tensors)
- "mock": Mock backend for single-process testing
"""

from __future__ import annotations
from typing import Union, Optional
from enum import Enum

from .comm_interface import CommBase


class CommBackendType(Enum):
    """Communication backend types."""
    MPI = "mpi"
    TORCH = "torch"
    MOCK = "mock"


def get_comm_backend(
    backend: Union[str, CommBackendType] = "mpi",
    **kwargs
) -> CommBase:
    """
    Factory function to create a communication backend.
    
    Args:
        backend: Backend type ("mpi", "torch", or "mock")
        **kwargs: Additional arguments passed to the backend constructor
            
            For MPI backend:
                - mpi_comm: MPI communicator (optional)
                - use_mpi: Whether to use real MPI (default True)
                
            For Torch backend:
                - torch_backend: Communication backend for torch ("nccl", "gloo", "mpi")
                - init_method: URL specifying how to initialize
                - world_size: Number of processes
                - rank: Rank of this process
                - auto_init: Whether to auto-initialize
                
    Returns:
        Communication backend instance (CommMPI, CommTorch, or Mock)
        
    Example:
        >>> # Create MPI backend for numpy arrays
        >>> mpi_comm = get_comm_backend("mpi")
        >>> 
        >>> # Create PyTorch distributed backend for tensors
        >>> torch_comm = get_comm_backend("torch", torch_backend="nccl")
        >>> 
        >>> # Create mock backend for testing
        >>> mock_comm = get_comm_backend("mock")
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        backend = backend.lower()
        if backend == "mpi":
            backend_type = CommBackendType.MPI
        elif backend == "torch" or backend == "pytorch":
            backend_type = CommBackendType.TORCH
        elif backend == "mock":
            backend_type = CommBackendType.MOCK
        else:
            raise ValueError(f"Unknown backend type: {backend}. "
                           f"Supported: 'mpi', 'torch', 'mock'")
    else:
        backend_type = backend
    
    if backend_type == CommBackendType.MPI:
        return _create_mpi_backend(**kwargs)
    elif backend_type == CommBackendType.TORCH:
        return _create_torch_backend(**kwargs)
    elif backend_type == CommBackendType.MOCK:
        return _create_mock_backend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _create_mpi_backend(
    mpi_comm=None,
    use_mpi: bool = True,
    **kwargs
) -> CommBase:
    """
    Create MPI communication backend.
    
    Args:
        mpi_comm: MPI communicator (optional)
        use_mpi: Whether to use real MPI backend
        
    Returns:
        CommMPI or MockCommMPI
    """
    from .comm_mpi import CommMPI, MockCommMPI
    
    if not use_mpi:
        return MockCommMPI()
    
    try:
        return CommMPI(mpi_comm=mpi_comm)
    except ImportError:
        print("Warning: mpi4py not available, falling back to MockCommMPI")
        return MockCommMPI()


def _create_torch_backend(
    torch_backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    node_rank: Optional[int] = None,
    num_nodes: Optional[int] = None,
    auto_init: bool = True,
    **kwargs
) -> CommBase:
    """
    Create PyTorch distributed communication backend.
    
    Args:
        torch_backend: Communication backend ("nccl", "gloo", "mpi")
        init_method: URL specifying how to initialize the process group
        world_size: Number of processes
        rank: Global rank of this process
        node_rank: Node rank / node index
        num_nodes: Number of nodes
        auto_init: Whether to auto-initialize
        
    Returns:
        CommTorch or MockCommTorch
    """
    from .comm_torch import CommTorch, MockCommTorch
    
    try:
        return CommTorch(
            torch_backend=torch_backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            node_rank=node_rank,
            num_nodes=num_nodes,
            auto_init=auto_init
        )
    except Exception as e:
        print(f"Warning: PyTorch distributed not available ({e}), "
              f"falling back to MockCommTorch")
        return MockCommTorch(rank=rank, world_size=world_size, node_rank=node_rank, num_nodes=num_nodes)


def _create_mock_backend(
    backend_type: str = "mpi", 
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    node_rank: Optional[int] = None,
    num_nodes: Optional[int] = None,
    **kwargs
) -> CommBase:
    """
    Create mock communication backend for testing.
    
    Args:
        backend_type: Which mock to create ("mpi" or "torch")
        rank: Simulated rank (default 0)
        world_size: Simulated world size (default 1)
        node_rank: Simulated node rank (default 0)
        num_nodes: Simulated number of nodes (default 1)
        
    Returns:
        MockCommMPI or MockCommTorch
    """
    if backend_type == "torch":
        from .comm_torch import MockCommTorch
        return MockCommTorch(rank=rank, world_size=world_size, node_rank=node_rank, num_nodes=num_nodes)
    else:
        from .comm_mpi import MockCommMPI
        return MockCommMPI(rank=rank, world_size=world_size, node_rank=node_rank, num_nodes=num_nodes)


def detect_best_backend() -> str:
    """
    Detect the best available communication backend.
    
    Returns:
        "torch" if PyTorch distributed is available and initialized,
        "mpi" if mpi4py is available,
        "mock" otherwise.
    """
    # Check PyTorch distributed first
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return "torch"
    except ImportError:
        pass
    
    # Check MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            return "mpi"
    except ImportError:
        pass
    
    # Check if running in a distributed environment via environment variables
    import os
    if os.environ.get("WORLD_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE"):
        # Likely in a distributed environment
        try:
            from mpi4py import MPI
            return "mpi"
        except ImportError:
            pass
        
        try:
            import torch.distributed
            return "torch"
        except ImportError:
            pass
    
    return "mock"


def get_auto_backend(**kwargs) -> CommBase:
    """
    Automatically detect and create the best available backend.
    
    Args:
        **kwargs: Additional arguments passed to backend constructor
        
    Returns:
        Communication backend instance
    """
    backend = detect_best_backend()
    return get_comm_backend(backend, **kwargs)


# Convenience functions for specific backends
def create_comm_mpi(**kwargs) -> CommBase:
    """Get MPI backend (shortcut for get_comm_backend('mpi'))."""
    return get_comm_backend("mpi", **kwargs)


def create_comm_torch(**kwargs) -> CommBase:
    """Get PyTorch distributed backend (shortcut for get_comm_backend('torch'))."""
    return get_comm_backend("torch", **kwargs)


def get_mock_backend(**kwargs) -> CommBase:
    """Get mock backend (shortcut for get_comm_backend('mock'))."""
    return get_comm_backend("mock", **kwargs)


# Backward compatibility aliases
get_mpi_backend = create_comm_mpi
get_torch_backend = create_comm_torch