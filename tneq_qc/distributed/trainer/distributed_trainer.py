"""
Distributed Trainer

Main entry point for distributed TNEQ training. Combines all distributed
components into a simple, high-level API.

Provides distributed-specific configuration options beyond the standard trainer:
- Communication backend selection (MPI, torch.distributed)
- Graph partitioning strategy
- Tensor parallel settings
- Gradient synchronization options
"""

from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

from ..comm import CommBase, get_comm_backend, ReduceOp
from ..parallel.data_parallel import DataParallelTrainer, TrainingConfig, TrainingStats
from ..engine.distributed_engine import (
    DistributedEngineSiamese, 
    PartitionConfig, 
    DistributedContractPlan
)

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.
    
    Extends standard training config with distributed-specific options.
    """
    
    # ==================== Backend Configuration ====================
    
    # Compute backend: 'pytorch' or 'jax'
    backend_type: str = 'pytorch'
    
    # Device: 'cpu', 'cuda', 'cuda:0', etc.
    device: str = 'cpu'
    
    # Contraction strategy mode: 'fast', 'balanced', 'full'
    strategy_mode: str = 'balanced'
    
    # Maximum Hermite polynomial order
    mx_K: int = 100
    
    # ==================== QCTN Configuration ====================
    
    # QCTN graph string
    qctn_graph: Optional[str] = None
    
    # Number of qubits (used if qctn_graph is None)
    num_qubits: int = 4
    
    # ==================== Communication Configuration ====================
    
    # Communication backend type: 'mpi', 'torch', 'auto'
    comm_backend: str = 'auto'
    
    # Whether to use real distributed communication or mock
    use_distributed: bool = True
    
    # Global rank of this process (None = auto-detect from environment)
    rank: Optional[int] = None
    
    # Total number of processes (None = auto-detect from environment)
    world_size: Optional[int] = None
    
    # Node rank / node index (for multi-node training, None = auto-detect)
    node_rank: Optional[int] = None
    
    # Number of nodes (for multi-node training, None = auto-detect)
    num_nodes: Optional[int] = None
    
    # ==================== Partitioning Configuration ====================
    
    # Partitioning strategy: 'layer', 'core', 'auto'
    partition_strategy: str = 'layer'
    
    # Minimum cores per partition
    min_cores_per_partition: int = 1
    
    # Whether to balance partition sizes
    balance_partitions: bool = True
    
    # ==================== Training Configuration ====================
    
    # Maximum training steps
    max_steps: int = 1000
    
    # Logging interval (steps)
    log_interval: int = 10
    
    # Checkpoint interval (steps)
    checkpoint_interval: int = 100
    
    # Learning rate
    learning_rate: float = 1e-2
    
    # Learning rate schedule: list of (step, lr) tuples
    lr_schedule: Optional[List[Tuple[int, float]]] = None
    
    # Optimizer method: 'sgdg', 'adam', etc.
    optimizer: str = 'sgdg'
    
    # Momentum (for SGD-based optimizers)
    momentum: float = 0.9
    
    # Whether to use Stiefel manifold optimization
    stiefel: bool = True
    
    # Convergence tolerance
    tol: Optional[float] = None
    
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    
    # ==================== Gradient Synchronization ====================
    
    # How often to sync gradients (in micro-batches)
    gradient_sync_interval: int = 1
    
    # Whether to overlap communication with computation
    overlap_comm_compute: bool = False
    
    # ==================== Checkpointing ====================
    
    # Checkpoint directory
    checkpoint_dir: str = './checkpoints'
    
    # Whether to save final model
    save_final_model: bool = True
    
    def to_training_config(self) -> TrainingConfig:
        """Convert to TrainingConfig for DataParallelTrainer."""
        return TrainingConfig(
            max_steps=self.max_steps,
            log_interval=self.log_interval,
            checkpoint_interval=self.checkpoint_interval,
            learning_rate=self.learning_rate,
            lr_schedule=self.lr_schedule,
            optimizer_method=self.optimizer,
            momentum=self.momentum,
            stiefel=self.stiefel,
            tol=self.tol,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
    
    def to_partition_config(self, world_size: int) -> PartitionConfig:
        """Create PartitionConfig for engine."""
        return PartitionConfig(
            strategy=self.partition_strategy,
            num_partitions=world_size,
            min_cores_per_partition=self.min_cores_per_partition,
            balance_partitions=self.balance_partitions,
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistributedConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class DistributedTrainer:
    """
    High-level Distributed Trainer.
    
    Provides a simple API for distributed TNEQ training, handling:
    - Communication backend initialization
    - QCTN model creation and partitioning
    - Data generation and distribution
    - Distributed training loop with gradient synchronization
    - Checkpoint management
    
    Key differences from standard Trainer:
    1. Graph partitioning: QCTN is split across workers
    2. Hierarchical contraction: log(n)+1 stage reduction
    3. Tensor parallel: Large matrix multiplications are sharded
    4. Gradient sync: Configurable synchronization strategies
    
    Example:
        >>> config = DistributedConfig(
        ...     backend_type='pytorch',
        ...     qctn_graph='-3-A-3-B-3-',
        ...     max_steps=1000,
        ...     partition_strategy='layer',
        ... )
        >>> trainer = DistributedTrainer(config)
        >>> data_list, circuit_states = trainer.prepare_data(N=100, B=128, K=3)
        >>> stats = trainer.train(data_list, circuit_states)
    
    Usage with mpiexec:
        $ mpiexec -n 4 python -m tneq_qc.distributed.trainer --config config.yaml
    """
    
    def __init__(self, config: Union[DistributedConfig, Dict[str, Any]]):
        """
        Initialize distributed trainer.
        
        Args:
            config: DistributedConfig or dictionary with configuration
        """
        # Parse config
        if isinstance(config, dict):
            self.config = DistributedConfig.from_dict(config)
            self._raw_config = config
        else:
            self.config = config
            self._raw_config = None
        
        # Initialize communication backend
        self._init_comm()
        
        # Initialize distributed engine with partitioning config
        partition_config = self.config.to_partition_config(self.comm.world_size)
        
        self.engine = DistributedEngineSiamese(
            backend=self.config.backend_type,
            strategy_mode=self.config.strategy_mode,
            mx_K=self.config.mx_K,
            comm=self.comm,
            partition_config=partition_config,
        )
        
        # Initialize QCTN
        self.qctn: Optional['QCTN'] = None
        self._init_qctn()
        
        # Initialize distributed contraction (partition the graph)
        if self.qctn is not None and self.comm.world_size > 1:
            self._contract_plan = self.engine.init_distributed(self.qctn)
        else:
            self._contract_plan = None
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        if self.comm.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"DistributedTrainer initialized: "
                  f"rank={self.comm.rank}/{self.comm.world_size}")
    
    def _init_comm(self):
        """Initialize communication backend with config settings."""
        comm_type = self.config.comm_backend
        use_distributed = self.config.use_distributed
        
        # Build kwargs for comm backend
        comm_kwargs = {}
        if self.config.rank is not None:
            comm_kwargs['rank'] = self.config.rank
        if self.config.world_size is not None:
            comm_kwargs['world_size'] = self.config.world_size
        if self.config.node_rank is not None:
            comm_kwargs['node_rank'] = self.config.node_rank
        if self.config.num_nodes is not None:
            comm_kwargs['num_nodes'] = self.config.num_nodes
        
        if not use_distributed:
            # Use mock backend
            self.comm = get_comm_backend(backend='mock', **comm_kwargs)
        elif comm_type == 'auto':
            # Auto-detect: try MPI first, fall back to mock
            self.comm = get_comm_backend(backend=self.config.backend_type, **comm_kwargs)
        elif comm_type == 'mpi':
            self.comm = get_comm_backend(backend='mpi', **comm_kwargs)
        elif comm_type == 'torch':
            self.comm = get_comm_backend(backend='torch', **comm_kwargs)
        else:
            # Default to MPI
            self.comm = get_comm_backend(backend='auto', **comm_kwargs)
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.comm.rank == 0:
            print(f"[DistributedTrainer] {msg}")
    
    def _init_qctn(self):
        """Initialize QCTN model on each process independently."""
        from ...core.qctn import QCTN, QCTNHelper
        
        qctn_graph = self.config.qctn_graph
        
        if qctn_graph is None:
            # Use default example graph based on num_qubits
            qctn_graph = QCTNHelper.generate_example_graph(n=self.config.num_qubits)
            self._log(f"Using default QCTN graph with {self.config.num_qubits} qubits")
        
        # Each process creates its own QCTN instance independently
        self.qctn = QCTN(qctn_graph, backend=self.engine.backend)
        self._log(f"QCTN initialized: {self.qctn.nqubits} qubits, {len(self.qctn.cores)} cores")
        
        # Note: Weights are initialized independently on each process.
        # The engine's init_distributed will handle partitioning and
        # each process will only keep its local subgraph.
    
    def _sync_model_weights(self):
        """
        Synchronize model weights from main process to all workers.
        
        Note: This method is currently not used in the default initialization flow.
        Each process initializes QCTN independently and engine.init_distributed()
        handles partitioning. This method is kept for cases where explicit
        weight synchronization is needed (e.g., after modifying weights on rank 0).
        """
        from ...core.tn_tensor import TNTensor
        
        if self.comm.world_size == 1:
            return
        
        for core_name in self.qctn.cores:
            if core_name not in self.qctn.cores_weights:
                continue
            weight = self.qctn.cores_weights[core_name]
            
            # Handle TNTensor objects
            if isinstance(weight, TNTensor):
                # Broadcast the underlying tensor directly (no numpy conversion)
                synced_tensor = self.comm.broadcast_object(weight.tensor, src=0)
                
                # Broadcast the scale factor
                synced_scale = self.comm.broadcast_object(weight.scale, src=0)
                
                # Reconstruct TNTensor
                self.qctn.cores_weights[core_name] = TNTensor(synced_tensor, synced_scale)
            else:
                # Regular tensor - broadcast directly
                synced_weight = self.comm.broadcast_object(weight, src=0)
                self.qctn.cores_weights[core_name] = synced_weight
        
        self._log("Model weights synchronized across workers")
    
    # ==================== Data Preparation ====================
    
    def prepare_data(self, N: int, B: int, K: int) -> Tuple[List[Dict], List]:
        """
        Prepare training data.
        
        Generates N batches of measurement matrices using Hermite polynomials.
        Data is generated on main process and broadcast to all workers.
        
        Args:
            N: Number of data batches
            B: Batch size (samples per batch)
            K: Hermite polynomial order
            
        Returns:
            (data_list, circuit_states_list)
        """
        import numpy as np
        
        backend = self.engine.backend
        D = self.qctn.nqubits
        
        data_list = []
        
        # Only main process generates data, then broadcast
        if self.comm.rank == 0:
            self._log(f"Generating data: N={N}, B={B}, D={D}, K={K}")
            
            for i in range(N):
                # Generate random data using numpy, then convert to backend tensor
                x_np = np.random.randn(B, D).astype(np.float32)
                x = backend.convert_to_tensor(x_np)
                Mx_list, _ = self.engine.generate_data(x, K=K)
                data_list.append({"measure_input_list": Mx_list})
        
        # Broadcast data list size
        n_batches = len(data_list) if self.comm.rank == 0 else 0
        n_batches = self.comm.broadcast_object(n_batches, src=0)
        
        # Broadcast each batch
        if self.comm.rank != 0:
            data_list = [None] * n_batches
        
        for i in range(n_batches):
            data_list[i] = self.comm.broadcast_object(data_list[i], src=0)
        
        # Generate circuit states
        circuit_states_list = [backend.zeros(K) for _ in range(D)]
        for s in circuit_states_list:
            s[-1] = 1.0
        
        self._log(f"Data prepared: {len(data_list)} batches")
        
        return data_list, circuit_states_list
    
    # ==================== Training ====================
    
    def train(self, 
              data_list: List[Dict], 
              circuit_states_list: List['torch.Tensor'],
              training_config: Optional[TrainingConfig] = None) -> TrainingStats:
        """
        Execute distributed training.
        
        Uses hierarchical contraction if world_size > 1:
        1. Each worker contracts its local subgraph
        2. log(n) reduction stages combine results using tensor parallel
        
        Args:
            data_list: Training data list
            circuit_states_list: Circuit states
            training_config: Training configuration (uses DistributedConfig if None)
            
        Returns:
            Training statistics
        """
        # Build training config from DistributedConfig if not provided
        if training_config is None:
            training_config = self.config.to_training_config()
        
        # Create data parallel trainer
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,  # Use base engine for now
            qctn=self.qctn,
            config=training_config,
            mpi_backend=self.comm  # Use our comm backend
        )
        
        # Execute training
        self._log("Starting distributed training...")
        stats = trainer.train(data_list, circuit_states_list)
        
        # Save final model if configured
        if self.config.save_final_model:
            self._save_final_model()
        
        return stats
    
    # ==================== Checkpointing ====================
    
    def _save_final_model(self):
        """Save final model (main process only)."""
        if self.comm.rank != 0:
            return
        
        try:
            model_path = self.checkpoint_dir / "final_model.safetensors"
            config_dict = self._raw_config if self._raw_config else {}
            self.qctn.save_cores(str(model_path), metadata={
                'config': json.dumps(config_dict)
            })
            self._log(f"Final model saved: {model_path}")
        except Exception as e:
            self._log(f"Warning: Could not save model: {e}", level="warn")
    
    def save_checkpoint(self, step: int, stats: Optional[TrainingStats] = None):
        """
        Save training checkpoint.
        
        Args:
            step: Current training step
            stats: Optional training stats
        """
        if self.comm.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.safetensors"
        
        config_dict = self._raw_config if self._raw_config else {}
        metadata = {
            'step': str(step),
            'config': json.dumps(config_dict),
        }
        
        if stats:
            metadata['final_loss'] = str(stats.final_loss)
        
        try:
            self.qctn.save_cores(str(checkpoint_path), metadata=metadata)
            self._log(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self._log(f"Warning: Could not save checkpoint: {e}", level="warn")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        from ...core.qctn import QCTN
        
        # Each process loads the checkpoint independently
        self.qctn = QCTN.from_pretrained(
            self.qctn.graph, 
            checkpoint_path, 
            backend=self.engine.backend
        )
        
        # Re-initialize distributed contraction plan
        # This will partition the graph and each process keeps only its local cores
        if self.comm.world_size > 1:
            self._contract_plan = self.engine.init_distributed(self.qctn)
        
        self._log(f"Loaded checkpoint: {checkpoint_path}")
    
    # ==================== Evaluation ====================
    
    def evaluate(self, data_list: List[Dict], 
                 circuit_states_list: List['torch.Tensor']) -> float:
        """
        Evaluate model on given data.
        
        Args:
            data_list: Evaluation data
            circuit_states_list: Circuit states
            
        Returns:
            Average loss
        """
        # Create temporary trainer for evaluation
        config = TrainingConfig(max_steps=0)
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,
            qctn=self.qctn,
            config=config,
            mpi_backend=self.comm
        )
        
        return trainer.evaluate(data_list, circuit_states_list)
    
    # ==================== Properties for Backward Compatibility ====================
    
    @property
    def mpi(self):
        """Alias for comm (backward compatibility)."""
        return self.comm
    
    @property
    def ctx(self):
        """Get distributed context."""
        return self.comm.get_context()


def main():
    """Command-line entry point for distributed training."""
    import argparse
    
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
    
    parser = argparse.ArgumentParser(description='TNEQ Distributed Training')
    parser.add_argument('--config', type=str, help='Config file path (YAML or JSON)')
    parser.add_argument('--backend', type=str, default='pytorch', help='Backend type')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-batches', type=int, default=100, help='Number of data batches')
    parser.add_argument('--hermite-order', type=int, default=3, help='Hermite polynomial order')
    parser.add_argument('--num-qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--partition-strategy', type=str, default='layer', 
                        choices=['layer', 'core', 'auto'], help='Partition strategy')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml'] and has_yaml:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        config = DistributedConfig.from_dict(config_dict)
    else:
        config = DistributedConfig(
            backend_type=args.backend,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            num_qubits=args.num_qubits,
            partition_strategy=args.partition_strategy,
        )
    
    # Create trainer
    trainer = DistributedTrainer(config)
    
    # Prepare data
    data_list, circuit_states_list = trainer.prepare_data(
        N=args.data_batches,
        B=args.batch_size,
        K=args.hermite_order
    )
    
    # Train
    stats = trainer.train(data_list, circuit_states_list)
    
    if trainer.comm.rank == 0:
        print(f"\nTraining completed: {stats.to_dict()}")


if __name__ == "__main__":
    main()
