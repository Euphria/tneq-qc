"""
PyTorch Distributed Communication Backend

Provides PyTorch distributed communication primitives for distributed training:
- Point-to-point communication (send, recv, isend, irecv)
- Collective operations (AllReduce, Broadcast, AllGather)
- Asynchronous communication support

This backend accepts PyTorch tensors for communication using torch.distributed.
"""

from __future__ import annotations
import torch
from typing import List, Optional, Union, Any, Tuple
import os

from .comm_interface import CommBase, ReduceOp, DistributedContext, AsyncHandle as AsyncHandleBase

# Lazy import torch.distributed
_dist = None


def _get_dist():
    """Lazy load torch.distributed to avoid import errors."""
    global _dist
    if _dist is None:
        try:
            import torch.distributed as dist
            _dist = dist
        except ImportError:
            raise ImportError(
                "torch.distributed is required for distributed training. "
                "Make sure PyTorch is installed with distributed support."
            )
    return _dist


def _reduce_op_to_torch(op: ReduceOp):
    """Convert ReduceOp to PyTorch ReduceOp."""
    dist = _get_dist()
    mapping = {
        ReduceOp.SUM: dist.ReduceOp.SUM,
        ReduceOp.AVG: dist.ReduceOp.SUM,  # AVG uses SUM then divide
        ReduceOp.MAX: dist.ReduceOp.MAX,
        ReduceOp.MIN: dist.ReduceOp.MIN,
        ReduceOp.PRODUCT: dist.ReduceOp.PRODUCT,
    }
    return mapping[op]


class TorchAsyncHandle(AsyncHandleBase):
    """Handle for asynchronous communication operations."""
    
    def __init__(self, work_handles: List, tensors: List[torch.Tensor], 
                 op: ReduceOp, world_size: int):
        """
        Initialize async handle.
        
        Args:
            work_handles: List of torch.distributed Work objects
            tensors: List of tensor buffers
            op: Reduction operation type
            world_size: Number of workers
        """
        self.work_handles = work_handles
        self.tensors = tensors
        self.op = op
        self.world_size = world_size
        self._completed = False
        self._result_tensors = None
    
    def wait(self) -> List[torch.Tensor]:
        """
        Wait for all communications to complete.
        
        Returns:
            List of result tensors
        """
        if self._completed:
            return self._result_tensors
        
        for work in self.work_handles:
            work.wait()
        
        result_tensors = []
        for tensor in self.tensors:
            if self.op == ReduceOp.AVG:
                tensor = tensor / self.world_size
            result_tensors.append(tensor)
        
        self._result_tensors = result_tensors
        self._completed = True
        return result_tensors
    
    def is_completed(self) -> bool:
        """Check if all communications are completed."""
        if self._completed:
            return True
        return all(work.is_completed() for work in self.work_handles)


class CommTorch(CommBase[torch.Tensor]):
    """
    PyTorch Distributed Communication Backend.
    
    Encapsulates PyTorch distributed communication primitives and provides 
    tensor-level communication interfaces for distributed training.
    
    Example:
        >>> comm = CommTorch()
        >>> if comm.is_main_process():
        ...     print("Hello from main process")
        >>> 
        >>> # AllReduce gradients
        >>> local_grad = torch.randn(10, 10)
        >>> global_grad = comm.allreduce(local_grad, op=ReduceOp.AVG)
    """
    
    def __init__(self, 
                 torch_backend: str = "nccl",
                 init_method: Optional[str] = None,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 node_rank: Optional[int] = None,
                 num_nodes: Optional[int] = None,
                 auto_init: bool = True):
        """
        Initialize PyTorch distributed backend.
        
        Args:
            torch_backend: Communication backend ("nccl", "gloo", "mpi")
            init_method: URL specifying how to initialize the process group
            world_size: Number of processes (auto-detected from env if None)
            rank: Global rank of this process (auto-detected from env if None)
            node_rank: Node rank / node index (auto-detected from env if None)
            num_nodes: Number of nodes (auto-detected from env if None)
            auto_init: Whether to auto-initialize if not already initialized
        """
        dist = _get_dist()
        
        # Check if already initialized
        if dist.is_initialized():
            self._initialized = True
        elif auto_init:
            # Try to get values from environment
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            if rank is None:
                rank = int(os.environ.get("RANK", 0))
            if init_method is None:
                init_method = os.environ.get("MASTER_ADDR", None)
                if init_method:
                    port = os.environ.get("MASTER_PORT", "29500")
                    init_method = f"tcp://{init_method}:{port}"
                else:
                    init_method = "env://"
            
            # Select appropriate backend
            if torch_backend == "nccl" and not torch.cuda.is_available():
                torch_backend = "gloo"
            
            if world_size > 1:
                dist.init_process_group(
                    backend=torch_backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank
                )
            self._initialized = dist.is_initialized()
        else:
            self._initialized = False
        
        if self._initialized:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank if rank is not None else 0
            self._world_size = world_size if world_size is not None else 1
        
        # Node info - priority: explicit parameter > environment variable > default
        if node_rank is not None:
            self._node_rank = node_rank
        else:
            self._node_rank = int(os.environ.get("NODE_RANK", 0))
        
        if num_nodes is not None:
            self._num_nodes = num_nodes
        else:
            self._num_nodes = int(os.environ.get("NNODES", 1))
        
        self._context = DistributedContext(
            world_size=self._world_size,
            rank=self._rank,
            node_rank=self._node_rank,
            num_nodes=self._num_nodes,
            is_main_process=(self._rank == 0),
            backend="torch"
        )
        
        self._backend_name = torch_backend
    
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
    
    def get_context(self) -> DistributedContext:
        """Get the distributed context."""
        return self._context
    
    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self._rank == 0
    
    def is_initialized(self) -> bool:
        """Check if distributed is initialized."""
        return self._initialized
    
    # ==================== Synchronization ====================
    
    def barrier(self) -> None:
        """Global synchronization barrier."""
        if self._initialized:
            dist = _get_dist()
            dist.barrier()
    
    # ==================== Broadcast Operations ====================
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast a tensor from source to all processes.
        
        Args:
            tensor: Tensor to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted tensor (same on all processes)
        """
        if not self._initialized:
            return tensor.clone()
        
        dist = _get_dist()
        
        # Make tensor contiguous
        tensor = tensor.contiguous()
        
        dist.broadcast(tensor, src=src)
        
        return tensor
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """
        Broadcast a Python object from source to all processes.
        
        Args:
            obj: Object to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted object (same on all processes)
        """
        if not self._initialized:
            return obj
        
        dist = _get_dist()
        
        object_list = [obj] if self._rank == src else [None]
        dist.broadcast_object_list(object_list, src=src)
        
        return object_list[0]
    
    # ==================== Reduce Operations ====================
    
    def allreduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        """
        AllReduce operation on a tensor.
        
        All processes contribute their tensor and receive the reduced result.
        
        Args:
            tensor: Input tensor
            op: Reduction operation type
            
        Returns:
            Reduced tensor (same on all processes)
        """
        if not self._initialized:
            return tensor.clone()
        
        dist = _get_dist()
        
        # Clone to avoid modifying the original
        result = tensor.clone().contiguous()
        
        dist.all_reduce(result, op=_reduce_op_to_torch(op))
        
        if op == ReduceOp.AVG:
            result = result / self._world_size
        
        return result
    
    def allreduce_inplace(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        """
        In-place AllReduce operation to reduce memory allocation.
        
        Args:
            tensor: Input/output tensor (will be modified)
            op: Reduction operation type
            
        Returns:
            The modified tensor
        """
        if not self._initialized:
            return tensor
        
        dist = _get_dist()
        
        dist.all_reduce(tensor, op=_reduce_op_to_torch(op))
        
        if op == ReduceOp.AVG:
            tensor.div_(self._world_size)
        
        return tensor
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM, 
                         device: Optional[torch.device] = None) -> float:
        """
        AllReduce a scalar value.
        
        Args:
            value: Local scalar value
            op: Reduction operation type
            device: Device to use for the tensor
            
        Returns:
            Reduced scalar value
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tensor = torch.tensor([value], device=device)
        result = self.allreduce(tensor, op)
        
        return float(result.item())
    
    # ==================== Gather Operations ====================
    
    def allgather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        AllGather operation: gather tensors from all processes.
        
        Args:
            tensor: Local tensor
            
        Returns:
            List of tensors from all processes
        """
        if not self._initialized:
            return [tensor.clone()]
        
        dist = _get_dist()
        
        tensor = tensor.contiguous()
        
        # Create output tensors
        gather_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        
        dist.all_gather(gather_list, tensor)
        
        return gather_list
    
    def reduce_scatter(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        """
        ReduceScatter operation (useful for tensor parallelism).
        
        Each process receives a portion of the reduced result.
        
        Args:
            tensor: Input tensor (size should be divisible by world_size)
            op: Reduction operation type
            
        Returns:
            This process's portion of the reduced result
        """
        if not self._initialized:
            chunk_size = tensor.numel() // self._world_size
            return tensor.flatten()[:chunk_size].clone()
        
        dist = _get_dist()
        
        # Calculate chunk size
        chunk_size = tensor.numel() // self._world_size
        
        # Create output tensor
        output = torch.zeros(chunk_size, dtype=tensor.dtype, device=tensor.device)
        
        # Split input into chunks
        input_list = list(tensor.flatten().chunk(self._world_size))
        
        dist.reduce_scatter(output, input_list, op=_reduce_op_to_torch(op))
        
        if op == ReduceOp.AVG:
            output = output / self._world_size
        
        return output
    
    # ==================== Point-to-Point Operations ====================
    
    def send(self, tensor: torch.Tensor, dest: int, tag: int = 0) -> None:
        """
        Send a tensor to a destination process.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
            tag: Message tag
        """
        if not self._initialized:
            return
        
        dist = _get_dist()
        tensor = tensor.contiguous()
        dist.send(tensor, dst=dest, tag=tag)
    
    def recv(self, src: int, tag: int = 0, **kwargs) -> torch.Tensor:
        """
        Receive a tensor from a source process.
        
        Args:
            src: Source rank
            tag: Message tag
            tensor: Buffer tensor to receive into (required)
            
        Returns:
            Received tensor
        """
        tensor = kwargs.get('tensor')
        if tensor is None:
            raise ValueError("tensor buffer is required for torch recv")
        
        if not self._initialized:
            return tensor
        
        dist = _get_dist()
        dist.recv(tensor, src=src, tag=tag)
        return tensor
    
    def isend(self, tensor: torch.Tensor, dest: int, tag: int = 0) -> Any:
        """
        Non-blocking send.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
            tag: Message tag
            
        Returns:
            Work handle
        """
        if not self._initialized:
            return None
        
        dist = _get_dist()
        tensor = tensor.contiguous()
        return dist.isend(tensor, dst=dest, tag=tag)
    
    def irecv(self, src: int, tag: int = 0, **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Non-blocking receive.
        
        Args:
            src: Source rank
            tag: Message tag
            tensor: Buffer tensor to receive into (required)
            
        Returns:
            Tuple of (tensor buffer, Work handle)
        """
        tensor = kwargs.get('tensor')
        if tensor is None:
            raise ValueError("tensor buffer is required for torch irecv")
        
        if not self._initialized:
            return tensor, None
        
        dist = _get_dist()
        work = dist.irecv(tensor, src=src, tag=tag)
        return tensor, work
    
    # ==================== Batch Operations ====================
    
    def allreduce_list(self, tensors: List[torch.Tensor], 
                          op: ReduceOp = ReduceOp.AVG) -> List[torch.Tensor]:
        """
        AllReduce a list of tensors (e.g., gradients).
        
        Args:
            tensors: List of tensors
            op: Reduction operation type
            
        Returns:
            List of reduced tensors
        """
        return [self.allreduce(t, op) for t in tensors]
    
    def allreduce_list_async(self, tensors: List[torch.Tensor], 
                              op: ReduceOp = ReduceOp.AVG) -> TorchAsyncHandle:
        """
        Asynchronous AllReduce for communication-computation overlap.
        
        Args:
            tensors: List of tensors
            op: Reduction operation type
            
        Returns:
            TorchAsyncHandle that can be waited on
        """
        if not self._initialized:
            return TorchAsyncHandle([], [t.clone() for t in tensors], op, self._world_size)
        
        dist = _get_dist()
        
        work_handles = []
        result_tensors = []
        
        for tensor in tensors:
            result = tensor.clone().contiguous()
            result_tensors.append(result)
            
            work = dist.all_reduce(result, op=_reduce_op_to_torch(op), async_op=True)
            work_handles.append(work)
        
        return TorchAsyncHandle(work_handles, result_tensors, op, self._world_size)
    
    # ==================== Cleanup ====================
    
    def destroy(self) -> None:
        """Destroy the process group."""
        if self._initialized:
            dist = _get_dist()
            dist.destroy_process_group()
            self._initialized = False


class MockCommTorch(CommBase[torch.Tensor]):
    """
    Mock PyTorch distributed backend for testing without distributed setup.
    
    Simulates single-process behavior.
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
            backend="torch"
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
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        return tensor.clone()
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj
    
    def allreduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        return tensor.clone()
    
    def allreduce_inplace(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        return tensor
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM,
                         device: Optional[torch.device] = None) -> float:
        return value
    
    def allgather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        return [tensor.clone()]
    
    def reduce_scatter(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        chunk_size = tensor.numel() // self._world_size
        return tensor.flatten()[:chunk_size].clone()
    
    def send(self, tensor: torch.Tensor, dest: int, tag: int = 0) -> None:
        pass
    
    def recv(self, src: int, tag: int = 0, **kwargs) -> torch.Tensor:
        tensor = kwargs.get('tensor')
        if tensor is None:
            return torch.zeros(1)
        return tensor
    
    def isend(self, tensor: torch.Tensor, dest: int, tag: int = 0) -> Any:
        return None
    
    def irecv(self, src: int, tag: int = 0, **kwargs) -> Tuple[torch.Tensor, Any]:
        tensor = kwargs.get('tensor')
        if tensor is None:
            tensor = torch.zeros(1)
        return tensor, None
    
    def allreduce_list(self, tensors: List[torch.Tensor], 
                          op: ReduceOp = ReduceOp.AVG) -> List[torch.Tensor]:
        return [t.clone() for t in tensors]
    
    def destroy(self) -> None:
        pass


def get_comm_torch(auto_init: bool = True, torch_backend: str = "nccl") -> Union[CommTorch, MockCommTorch]:
    """
    Get the appropriate PyTorch distributed backend.
    
    Args:
        auto_init: Whether to auto-initialize distributed
        torch_backend: Communication backend ("nccl", "gloo", "mpi")
        
    Returns:
        CommTorch or MockCommTorch
    """
    try:
        return CommTorch(torch_backend=torch_backend, auto_init=auto_init)
    except Exception as e:
        print(f"Warning: PyTorch distributed not available ({e}), falling back to MockCommTorch")
        return MockCommTorch()


# Backward compatibility aliases
TorchBackend = CommTorch
MockTorchBackend = MockCommTorch
get_torch_backend = get_comm_torch
AsyncHandle = TorchAsyncHandle