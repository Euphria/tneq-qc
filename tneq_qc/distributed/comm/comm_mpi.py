"""
MPI Communication Backend for NumPy Arrays

Provides MPI-based communication primitives for distributed training:
- Point-to-point communication (send, recv, isend, irecv)
- Collective operations (AllReduce, Broadcast, AllGather)
- Asynchronous communication support

This backend only accepts numpy arrays for communication.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Union, Any, Tuple

from .comm_interface import CommBase, ReduceOp, DistributedContext, AsyncHandle as AsyncHandleBase

# Lazy import mpi4py to allow module to be imported without MPI
_MPI = None
_MPI_COMM_WORLD = None


def _get_mpi():
    """Lazy load MPI to avoid import errors when MPI is not installed."""
    global _MPI, _MPI_COMM_WORLD
    if _MPI is None:
        try:
            from mpi4py import MPI
            _MPI = MPI
            _MPI_COMM_WORLD = MPI.COMM_WORLD
        except ImportError:
            raise ImportError(
                "mpi4py is required for distributed training. "
                "Install with: pip install mpi4py"
            )
    return _MPI, _MPI_COMM_WORLD


def _reduce_op_to_mpi(op: ReduceOp):
    """Convert ReduceOp to MPI operation."""
    MPI, _ = _get_mpi()
    mapping = {
        ReduceOp.SUM: MPI.SUM,
        ReduceOp.AVG: MPI.SUM,  # AVG uses SUM then divide
        ReduceOp.MAX: MPI.MAX,
        ReduceOp.MIN: MPI.MIN,
        ReduceOp.PRODUCT: MPI.PROD,
    }
    return mapping[op]


class MPIAsyncHandle(AsyncHandleBase):
    """Handle for asynchronous communication operations."""
    
    def __init__(self, requests: List, results: List[np.ndarray], 
                 op: ReduceOp, world_size: int):
        """
        Initialize async handle.
        
        Args:
            requests: List of MPI Request objects
            results: List of result buffers
            op: Reduction operation type
            world_size: Number of workers
        """
        self.requests = requests
        self.results = results
        self.op = op
        self.world_size = world_size
        self._completed = False
        self._arrays = None
    
    def wait(self) -> List[np.ndarray]:
        """
        Wait for all communications to complete.
        
        Returns:
            List of result numpy arrays
        """
        if self._completed:
            return self._arrays
        
        MPI, _ = _get_mpi()
        
        MPI.Request.Waitall(self.requests)
        
        arrays = []
        for result in self.results:
            if self.op == ReduceOp.AVG:
                result = result / self.world_size
            arrays.append(result.copy())
        
        self._arrays = arrays
        self._completed = True
        return arrays
    
    def is_completed(self) -> bool:
        """Check if all communications are completed."""
        if self._completed:
            return True
        return all(req.Test() for req in self.requests)


class CommMPI(CommBase[np.ndarray]):
    """
    MPI Communication Backend for NumPy Arrays.
    
    Encapsulates MPI communication primitives and provides numpy array-level
    communication interfaces for distributed training.
    
    Example:
        >>> comm = CommMPI()
        >>> if comm.is_main_process():
        ...     print("Hello from main process")
        >>> 
        >>> # AllReduce gradients
        >>> local_grad = np.random.randn(10, 10)
        >>> global_grad = comm.allreduce(local_grad, op=ReduceOp.AVG)
    """
    
    def __init__(self, mpi_comm=None):
        """
        Initialize MPI backend.
        
        Args:
            mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        """
        MPI, COMM_WORLD = _get_mpi()
        
        self._comm = mpi_comm if mpi_comm is not None else COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._world_size = self._comm.Get_size()
        
        # Node info - try to get from environment, default to single node
        import os
        self._node_rank = int(os.environ.get('NODE_RANK', 0))
        self._num_nodes = int(os.environ.get('NNODES', 1))
        
        self._context = DistributedContext(
            world_size=self._world_size,
            rank=self._rank,
            node_rank=self._node_rank,
            num_nodes=self._num_nodes,
            is_main_process=(self._rank == 0),
            backend="mpi"
        )
        
        self._initialized = True
    
    # ==================== Context Properties ====================
    
    @property
    def rank(self) -> int:
        """Get the rank of the current process."""
        return self._rank
    
    @property
    def world_size(self) -> int:
        """Get the total number of processes."""
        return self._world_size
    
    @property
    def node_rank(self) -> int:
        """Get the node rank."""
        return self._node_rank
    
    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes."""
        return self._num_nodes
    
    @property
    def comm(self):
        """Get the MPI communicator."""
        return self._comm
    
    def get_context(self) -> DistributedContext:
        """Get the distributed context."""
        return self._context
    
    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self._rank == 0
    
    def is_initialized(self) -> bool:
        """Check if the distributed backend is initialized."""
        return self._initialized
    
    # ==================== Synchronization ====================
    
    def barrier(self) -> None:
        """Global synchronization barrier."""
        self._comm.Barrier()
    
    # ==================== Broadcast Operations ====================
    
    def broadcast(self, data: np.ndarray, src: int = 0) -> np.ndarray:
        """
        Broadcast a numpy array from source to all processes.
        
        Args:
            data: Numpy array to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted numpy array (same on all processes)
        """
        # Broadcast shape and dtype first
        if self._rank == src:
            shape = data.shape
            dtype_str = str(data.dtype)
        else:
            shape = None
            dtype_str = None
        
        shape = self._comm.bcast(shape, root=src)
        dtype_str = self._comm.bcast(dtype_str, root=src)
        dtype = np.dtype(dtype_str)
        
        if self._rank != src:
            data = np.empty(shape, dtype=dtype)
        else:
            data = np.ascontiguousarray(data)
        
        self._comm.Bcast(data, root=src)
        
        return data.copy()
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """
        Broadcast a Python object from source to all processes.
        
        Args:
            obj: Object to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted object (same on all processes)
        """
        return self._comm.bcast(obj, root=src)
    
    # ==================== Reduce Operations ====================
    
    def allreduce(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        """
        AllReduce operation on a numpy array.
        
        All processes contribute their array and receive the reduced result.
        
        Args:
            data: Input numpy array
            op: Reduction operation type
            
        Returns:
            Reduced numpy array (same on all processes)
        """
        data = np.ascontiguousarray(data)
        result = np.zeros_like(data)
        
        self._comm.Allreduce(data, result, op=_reduce_op_to_mpi(op))
        
        if op == ReduceOp.AVG:
            result = result / self._world_size
        
        return result
    
    def allreduce_inplace(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        """
        In-place AllReduce operation to reduce memory allocation.
        
        Args:
            data: Input/output numpy array (will be modified)
            op: Reduction operation type
            
        Returns:
            The modified input array
        """
        MPI, _ = _get_mpi()
        
        data = np.ascontiguousarray(data)
        
        self._comm.Allreduce(MPI.IN_PLACE, data, op=_reduce_op_to_mpi(op))
        
        if op == ReduceOp.AVG:
            data /= self._world_size
        
        return data
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM) -> float:
        """
        AllReduce a scalar value.
        
        Args:
            value: Local scalar value
            op: Reduction operation type
            
        Returns:
            Reduced scalar value
        """
        local = np.array([value])
        result = np.zeros(1)
        
        self._comm.Allreduce(local, result, op=_reduce_op_to_mpi(op))
        
        if op == ReduceOp.AVG:
            result[0] /= self._world_size
        
        return float(result[0])
    
    # ==================== Gather Operations ====================
    
    def allgather(self, data: np.ndarray) -> List[np.ndarray]:
        """
        AllGather operation: gather numpy arrays from all processes.
        
        Args:
            data: Local numpy array
            
        Returns:
            List of numpy arrays from all processes
        """
        data = np.ascontiguousarray(data)
        all_data = self._comm.allgather(data)
        return [d.copy() for d in all_data]
    
    def reduce_scatter(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        """
        ReduceScatter operation (useful for tensor parallelism).
        
        Each process receives a portion of the reduced result.
        
        Args:
            data: Input numpy array (size should be divisible by world_size)
            op: Reduction operation type
            
        Returns:
            This process's portion of the reduced result
        """
        data = np.ascontiguousarray(data)
        chunk_size = data.size // self._world_size
        
        result = np.zeros(chunk_size, dtype=data.dtype)
        
        self._comm.Reduce_scatter(data, result, op=_reduce_op_to_mpi(op))
        
        if op == ReduceOp.AVG:
            result = result / self._world_size
        
        return result
    
    # ==================== Point-to-Point Operations ====================
    
    def send(self, data: np.ndarray, dest: int, tag: int = 0) -> None:
        """
        Send a numpy array to a destination process.
        
        Args:
            data: Numpy array to send
            dest: Destination rank
            tag: Message tag
        """
        data = np.ascontiguousarray(data)
        self._comm.Send(data, dest=dest, tag=tag)
    
    def recv(self, src: int, tag: int = 0, **kwargs) -> np.ndarray:
        """
        Receive a numpy array from a source process.
        
        Args:
            src: Source rank
            tag: Message tag
            shape: Expected array shape (required)
            dtype: Expected dtype (optional, default float64)
            
        Returns:
            Received numpy array
        """
        shape = kwargs.get('shape')
        dtype = kwargs.get('dtype', np.float64)
        
        if shape is None:
            raise ValueError("shape is required for MPI recv")
        
        data = np.empty(shape, dtype=dtype)
        self._comm.Recv(data, source=src, tag=tag)
        return data
    
    def isend(self, data: np.ndarray, dest: int, tag: int = 0) -> Any:
        """
        Non-blocking send.
        
        Args:
            data: Numpy array to send
            dest: Destination rank
            tag: Message tag
            
        Returns:
            MPI Request object
        """
        data = np.ascontiguousarray(data)
        return self._comm.Isend(data, dest=dest, tag=tag)
    
    def irecv(self, src: int, tag: int = 0, **kwargs) -> Tuple[np.ndarray, Any]:
        """
        Non-blocking receive.
        
        Args:
            src: Source rank
            tag: Message tag
            shape: Expected array shape (required)
            dtype: Expected dtype (optional, default float64)
            
        Returns:
            Tuple of (buffer, MPI Request)
        """
        shape = kwargs.get('shape')
        dtype = kwargs.get('dtype', np.float64)
        
        if shape is None:
            raise ValueError("shape is required for MPI irecv")
        
        data = np.empty(shape, dtype=dtype)
        req = self._comm.Irecv(data, source=src, tag=tag)
        return data, req
    
    # ==================== Batch Operations ====================
    
    def allreduce_list(self, data_list: List[np.ndarray], 
                       op: ReduceOp = ReduceOp.AVG) -> List[np.ndarray]:
        """
        AllReduce a list of numpy arrays (e.g., gradients).
        
        Args:
            data_list: List of numpy arrays
            op: Reduction operation type
            
        Returns:
            List of reduced numpy arrays
        """
        return [self.allreduce(arr, op) for arr in data_list]
    
    def allreduce_list_async(self, data_list: List[np.ndarray], 
                              op: ReduceOp = ReduceOp.AVG) -> MPIAsyncHandle:
        """
        Asynchronous AllReduce for communication-computation overlap.
        
        Args:
            data_list: List of numpy arrays
            op: Reduction operation type
            
        Returns:
            MPIAsyncHandle that can be waited on
        """
        requests = []
        results = []
        
        for arr in data_list:
            data = np.ascontiguousarray(arr)
            result = np.zeros_like(data)
            results.append(result)
            
            req = self._comm.Iallreduce(data, result, op=_reduce_op_to_mpi(op))
            requests.append(req)
        
        return MPIAsyncHandle(requests, results, op, self._world_size)


class MockCommMPI(CommBase[np.ndarray]):
    """
    Mock MPI backend for testing without MPI.
    
    Simulates single-process MPI behavior with numpy arrays.
    Can be configured with custom rank/world_size for testing.
    """
    
    def __init__(
        self, 
        rank: Optional[int] = None, 
        world_size: Optional[int] = None,
        node_rank: Optional[int] = None,
        num_nodes: Optional[int] = None,
    ):
        self._rank = rank if rank is not None else 0
        self._world_size = world_size if world_size is not None else 1
        self._node_rank = node_rank if node_rank is not None else 0
        self._num_nodes = num_nodes if num_nodes is not None else 1
        self._context = DistributedContext(
            world_size=self._world_size,
            rank=self._rank,
            node_rank=self._node_rank,
            num_nodes=self._num_nodes,
            is_main_process=(self._rank == 0),
            backend="mpi"
        )
        self._initialized = False
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def world_size(self) -> int:
        return self._world_size
    
    @property
    def node_rank(self) -> int:
        return self._node_rank
    
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    def get_context(self) -> DistributedContext:
        return self._context
    
    def is_main_process(self) -> bool:
        return self._rank == 0
    
    def is_initialized(self) -> bool:
        return False
    
    def barrier(self) -> None:
        pass
    
    def broadcast(self, data: np.ndarray, src: int = 0) -> np.ndarray:
        return data.copy()
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj
    
    def allreduce(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        return data.copy()
    
    def allreduce_inplace(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        return data
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM) -> float:
        return value
    
    def allgather(self, data: np.ndarray) -> List[np.ndarray]:
        return [data.copy()]
    
    def reduce_scatter(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        chunk_size = data.size // self._world_size
        return data.flatten()[:chunk_size].copy()
    
    def send(self, data: np.ndarray, dest: int, tag: int = 0) -> None:
        pass
    
    def recv(self, src: int, tag: int = 0, **kwargs) -> np.ndarray:
        shape = kwargs.get('shape', (1,))
        dtype = kwargs.get('dtype', np.float64)
        return np.zeros(shape, dtype=dtype)
    
    def isend(self, data: np.ndarray, dest: int, tag: int = 0) -> Any:
        return None
    
    def irecv(self, src: int, tag: int = 0, **kwargs) -> Tuple[np.ndarray, Any]:
        shape = kwargs.get('shape', (1,))
        dtype = kwargs.get('dtype', np.float64)
        return np.zeros(shape, dtype=dtype), None
    
    def allreduce_list(self, data_list: List[np.ndarray], 
                       op: ReduceOp = ReduceOp.AVG) -> List[np.ndarray]:
        return [arr.copy() for arr in data_list]


def get_comm_mpi(use_mpi: bool = True) -> Union[CommMPI, MockCommMPI]:
    """
    Get the appropriate MPI backend based on environment.
    
    Args:
        use_mpi: Whether to use real MPI backend
        
    Returns:
        CommMPI or MockCommMPI
    """
    if use_mpi:
        try:
            return CommMPI()
        except ImportError:
            print("Warning: mpi4py not available, falling back to MockCommMPI")
            return MockCommMPI()
    else:
        return MockCommMPI()


# Backward compatibility aliases
MPIBackend = CommMPI
MockMPIBackend = MockCommMPI
get_mpi_backend = get_comm_mpi
AsyncHandle = MPIAsyncHandle