"""
Communication Interface for Distributed Training

Defines the abstract base class CommBase that all communication backends must implement.
Provides a unified interface for:
- Point-to-point communication (send, recv, isend, irecv)
- Collective operations (allreduce, broadcast, allgather, reduce_scatter)
- Synchronization primitives (barrier)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Any, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Type variable for data types (np.ndarray or torch.Tensor)
T = TypeVar('T')


class ReduceOp(Enum):
    """Reduction operation types for collective communications."""
    SUM = "SUM"
    AVG = "AVG"  # Custom: SUM then divide by world_size
    MAX = "MAX"
    MIN = "MIN"
    PRODUCT = "PRODUCT"


@dataclass
class DistributedContext:
    """Distributed computing context information."""
    world_size: int
    rank: int
    node_rank: int
    num_nodes: int
    is_main_process: bool
    backend: str = "unknown"
    
    def __repr__(self):
        return f"DistributedContext(rank={self.rank}/{self.world_size}, node={self.node_rank}/{self.num_nodes}, main={self.is_main_process}, backend={self.backend})"


class CommBase(ABC, Generic[T]):
    """
    Abstract base class for communication backends.
    
    This interface defines all communication primitives that backends must implement.
    Backends can work with different data types (numpy arrays, torch tensors, etc.)
    
    Type parameter T represents the data type used by the backend:
    - For MPI backend: numpy.ndarray
    - For Torch backend: torch.Tensor
    
    Example:
        >>> class MyBackend(CommBase[np.ndarray]):
        ...     def allreduce(self, data, op=ReduceOp.SUM):
        ...         # Implementation for numpy arrays
        ...         pass
    """
    
    # ==================== Context Properties ====================
    
    @property
    @abstractmethod
    def rank(self) -> int:
        """Get the rank of the current process."""
        pass
    
    @property
    @abstractmethod
    def world_size(self) -> int:
        """Get the total number of processes."""
        pass
    
    @property
    @abstractmethod
    def node_rank(self) -> int:
        """Get the node rank (index of this node)."""
        pass
    
    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Get the total number of nodes."""
        pass
    
    @abstractmethod
    def get_context(self) -> DistributedContext:
        """Get the distributed context."""
        pass
    
    @abstractmethod
    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process."""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the distributed backend is initialized."""
        pass
    
    # ==================== Synchronization ====================
    
    @abstractmethod
    def barrier(self) -> None:
        """
        Global synchronization barrier.
        
        Blocks until all processes reach this point.
        """
        pass
    
    # ==================== Broadcast Operations ====================
    
    @abstractmethod
    def broadcast(self, data: T, src: int = 0) -> T:
        """
        Broadcast data from source to all processes.
        
        Args:
            data: Data to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted data (same on all processes)
        """
        pass
    
    @abstractmethod
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """
        Broadcast a Python object from source to all processes.
        
        Uses pickle serialization for arbitrary objects.
        
        Args:
            obj: Object to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted object (same on all processes)
        """
        pass
    
    # ==================== Reduce Operations ====================
    
    @abstractmethod
    def allreduce(self, data: T, op: ReduceOp = ReduceOp.SUM) -> T:
        """
        AllReduce operation on data.
        
        All processes contribute their data and receive the reduced result.
        
        Args:
            data: Input data
            op: Reduction operation type
            
        Returns:
            Reduced data (same on all processes)
        """
        pass
    
    @abstractmethod
    def allreduce_inplace(self, data: T, op: ReduceOp = ReduceOp.SUM) -> T:
        """
        In-place AllReduce operation.
        
        Modifies data in place to reduce memory allocation.
        
        Args:
            data: Input/output data (will be modified)
            op: Reduction operation type
            
        Returns:
            The modified input data
        """
        pass
    
    @abstractmethod
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM) -> float:
        """
        AllReduce a scalar value.
        
        Args:
            value: Local scalar value
            op: Reduction operation type
            
        Returns:
            Reduced scalar value
        """
        pass
    
    # ==================== Gather Operations ====================
    
    @abstractmethod
    def allgather(self, data: T) -> List[T]:
        """
        AllGather operation: gather data from all processes.
        
        Args:
            data: Local data
            
        Returns:
            List of data from all processes (ordered by rank)
        """
        pass
    
    @abstractmethod
    def reduce_scatter(self, data: T, op: ReduceOp = ReduceOp.SUM) -> T:
        """
        ReduceScatter operation.
        
        Reduces data across all processes and scatters the result.
        Each process receives a portion of the reduced result.
        
        Args:
            data: Input data (size should be divisible by world_size)
            op: Reduction operation type
            
        Returns:
            This process's portion of the reduced result
        """
        pass
    
    # ==================== Point-to-Point Operations ====================
    
    @abstractmethod
    def send(self, data: T, dest: int, tag: int = 0) -> None:
        """
        Blocking send.
        
        Sends data to destination process. Blocks until data is sent.
        
        Args:
            data: Data to send
            dest: Destination rank
            tag: Message tag for matching
        """
        pass
    
    @abstractmethod
    def recv(self, src: int, tag: int = 0, **kwargs) -> T:
        """
        Blocking receive.
        
        Receives data from source process. Blocks until data is received.
        
        Args:
            src: Source rank
            tag: Message tag for matching
            **kwargs: Backend-specific arguments (e.g., shape, dtype for MPI)
            
        Returns:
            Received data
        """
        pass
    
    @abstractmethod
    def isend(self, data: T, dest: int, tag: int = 0) -> Any:
        """
        Non-blocking send.
        
        Initiates send and returns immediately with a handle.
        
        Args:
            data: Data to send
            dest: Destination rank
            tag: Message tag for matching
            
        Returns:
            Request/Work handle that can be waited on
        """
        pass
    
    @abstractmethod
    def irecv(self, src: int, tag: int = 0, **kwargs) -> Tuple[T, Any]:
        """
        Non-blocking receive.
        
        Initiates receive and returns immediately with buffer and handle.
        
        Args:
            src: Source rank
            tag: Message tag for matching
            **kwargs: Backend-specific arguments (e.g., shape, dtype for MPI)
            
        Returns:
            Tuple of (buffer for received data, request/work handle)
        """
        pass
    
    # ==================== Batch Operations ====================
    
    def allreduce_list(self, data_list: List[T], op: ReduceOp = ReduceOp.AVG) -> List[T]:
        """
        AllReduce a list of data (e.g., gradients).
        
        Default implementation calls allreduce on each item.
        Subclasses may override for more efficient batch operations.
        
        Args:
            data_list: List of data
            op: Reduction operation type
            
        Returns:
            List of reduced data
        """
        return [self.allreduce(d, op) for d in data_list]
    
    # ==================== Cleanup ====================
    
    def destroy(self) -> None:
        """
        Cleanup and destroy the communication backend.
        
        Should be called when distributed training is complete.
        Default implementation does nothing.
        """
        pass


class AsyncHandle(ABC):
    """Abstract base class for asynchronous communication handles."""
    
    @abstractmethod
    def wait(self) -> Any:
        """
        Wait for the communication to complete.
        
        Returns:
            Result of the communication (if applicable)
        """
        pass
    
    @abstractmethod
    def is_completed(self) -> bool:
        """
        Check if the communication is completed.
        
        Returns:
            True if completed, False otherwise
        """
        pass
