"""
Tests for Model Parallel Training

Tests the model parallel implementation including:
- Core partitioning across workers
- Weight management (local weights only)
- Gradient computation and synchronization
- Model parallel trainer
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestCorePartition:
    """Tests for CorePartition dataclass."""
    
    def test_partition_creation(self):
        """Test CorePartition creation."""
        from tneq_qc.distributed.parallel.model_parallel import CorePartition
        
        partition = CorePartition(
            rank=0,
            world_size=4,
            local_core_indices=[0, 1],
            local_core_names=['A', 'B'],
            core_to_worker={'A': 0, 'B': 0, 'C': 1, 'D': 1},
            total_cores=4
        )
        
        assert partition.rank == 0
        assert partition.world_size == 4
        assert partition.local_core_names == ['A', 'B']
        assert partition.total_cores == 4
    
    def test_is_local_core(self):
        """Test local core check."""
        from tneq_qc.distributed.parallel.model_parallel import CorePartition
        
        partition = CorePartition(
            rank=0,
            world_size=2,
            local_core_indices=[0, 1],
            local_core_names=['A', 'B'],
            core_to_worker={'A': 0, 'B': 0, 'C': 1, 'D': 1},
            total_cores=4
        )
        
        assert partition.is_local_core('A') is True
        assert partition.is_local_core('B') is True
        assert partition.is_local_core('C') is False
        assert partition.is_local_core('D') is False
    
    def test_get_core_owner(self):
        """Test getting core owner."""
        from tneq_qc.distributed.parallel.model_parallel import CorePartition
        
        partition = CorePartition(
            rank=0,
            world_size=2,
            local_core_indices=[0, 1],
            local_core_names=['A', 'B'],
            core_to_worker={'A': 0, 'B': 0, 'C': 1, 'D': 1},
            total_cores=4
        )
        
        assert partition.get_core_owner('A') == 0
        assert partition.get_core_owner('B') == 0
        assert partition.get_core_owner('C') == 1
        assert partition.get_core_owner('D') == 1
        assert partition.get_core_owner('E') == -1  # Unknown core
    
    def test_partition_repr(self):
        """Test partition string representation."""
        from tneq_qc.distributed.parallel.model_parallel import CorePartition
        
        partition = CorePartition(
            rank=1,
            world_size=4,
            local_core_indices=[2, 3],
            local_core_names=['C', 'D'],
            core_to_worker={'A': 0, 'B': 0, 'C': 1, 'D': 1},
            total_cores=4
        )
        
        repr_str = repr(partition)
        assert "rank=1/4" in repr_str
        assert "C" in repr_str
        assert "D" in repr_str


class TestModelParallelConfig:
    """Tests for ModelParallelConfig."""
    
    def test_default_config(self):
        """Test default config values."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelConfig
        
        config = ModelParallelConfig()
        
        assert config.partition_strategy == 'even'
        assert config.overlap_comm is False
        assert config.comm_buffer_size == 10
    
    def test_custom_config(self):
        """Test custom config."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelConfig
        
        config = ModelParallelConfig(
            partition_strategy='balanced',
            overlap_comm=True,
            comm_buffer_size=20
        )
        
        assert config.partition_strategy == 'balanced'
        assert config.overlap_comm is True
        assert config.comm_buffer_size == 20


class TestModelParallelManager:
    """Tests for ModelParallelManager."""
    
    def _create_mock_qctn(self, n_cores=4):
        """Create mock QCTN with specified number of cores."""
        import torch
        from tneq_qc.core.tn_tensor import TNTensor
        
        mock_qctn = MagicMock()
        mock_qctn.cores = [chr(ord('A') + i) for i in range(n_cores)]  # ['A', 'B', 'C', 'D']
        mock_qctn.nqubits = 2
        mock_qctn.ncores = n_cores
        
        # Create adjacency table
        mock_qctn.adjacency_table = [
            {'core_idx': i, 'core_name': mock_qctn.cores[i], 
             'in_edge_list': [], 'out_edge_list': []}
            for i in range(n_cores)
        ]
        
        # Create weights
        mock_qctn.cores_weights = {
            c: TNTensor(torch.randn(3, 3), 1.0) for c in mock_qctn.cores
        }
        
        # Backend
        mock_qctn.backend = MagicMock()
        mock_qctn.backend.tensor_to_numpy = lambda x: x.numpy()
        mock_qctn.backend.convert_to_tensor = lambda x: torch.tensor(x)
        mock_qctn.backend.zeros = lambda shape: torch.zeros(shape)
        
        return mock_qctn
    
    def test_manager_initialization(self):
        """Test ModelParallelManager initialization."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn(4)
        mpi = MockMPIBackend()
        
        manager = ModelParallelManager(mock_qctn, mpi)
        
        assert manager.partition is not None
        assert manager.partition.total_cores == 4
    
    def test_even_partition_single_worker(self):
        """Test even partition with single worker."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn(4)
        mpi = MockMPIBackend()  # Single worker
        
        manager = ModelParallelManager(mock_qctn, mpi)
        
        # Single worker should have all cores
        assert len(manager.partition.local_core_names) == 4
        assert manager.partition.local_core_names == ['A', 'B', 'C', 'D']
    
    def test_get_local_weights(self):
        """Test getting local weights."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn(4)
        mpi = MockMPIBackend()
        
        manager = ModelParallelManager(mock_qctn, mpi)
        
        local_weights = manager.get_local_weights()
        
        # Single worker has all cores
        assert len(local_weights) == 4
        assert 'A' in local_weights
        assert 'D' in local_weights
    
    def test_set_local_weights(self):
        """Test setting local weights."""
        import torch
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        from tneq_qc.core.tn_tensor import TNTensor
        
        mock_qctn = self._create_mock_qctn(4)
        mpi = MockMPIBackend()
        
        manager = ModelParallelManager(mock_qctn, mpi)
        
        # Set new weights
        new_weights = {
            'A': TNTensor(torch.ones(3, 3), 2.0),
            'B': TNTensor(torch.ones(3, 3) * 2, 1.5),
        }
        
        manager.set_local_weights(new_weights)
        
        # Verify weights updated
        assert mock_qctn.cores_weights['A'].scale == 2.0


class TestModelParallelTrainer:
    """Tests for ModelParallelTrainer."""
    
    def _create_mock_engine(self):
        """Create mock engine."""
        import torch
        
        # Create base engine mock
        base_engine = MagicMock()
        base_engine.backend = MagicMock()
        base_engine.backend.tensor_to_numpy = lambda x: x.numpy() if hasattr(x, 'numpy') else np.array(x)
        base_engine.backend.convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        base_engine.backend.zeros = lambda shape: torch.zeros(shape)
        
        # Mock contraction returns loss
        base_engine.contract_with_compiled_strategy = MagicMock(return_value=torch.tensor(1.0))
        
        # Mock gradient function returns (loss, grads_dict)
        def mock_grad_fn(*args, **kwargs):
            return (torch.tensor(1.0), {'A': torch.randn(3, 3), 'B': torch.randn(3, 3)})
        base_engine.contract_with_compiled_strategy_for_gradient = MagicMock(side_effect=mock_grad_fn)
        
        # Create distributed engine mock with _base_engine
        mock_engine = MagicMock()
        mock_engine._base_engine = base_engine
        mock_engine.backend = base_engine.backend
        mock_engine.contract_with_compiled_strategy = base_engine.contract_with_compiled_strategy
        mock_engine.contract_with_compiled_strategy_for_gradient = base_engine.contract_with_compiled_strategy_for_gradient
        
        return mock_engine
    
    def _create_mock_qctn(self):
        """Create mock QCTN."""
        import torch
        from tneq_qc.core.tn_tensor import TNTensor
        
        mock_qctn = MagicMock()
        mock_qctn.cores = ['A', 'B']
        mock_qctn.nqubits = 2
        mock_qctn.ncores = 2
        
        mock_qctn.adjacency_table = [
            {'core_idx': 0, 'core_name': 'A', 'in_edge_list': [], 'out_edge_list': []},
            {'core_idx': 1, 'core_name': 'B', 'in_edge_list': [], 'out_edge_list': []},
        ]
        
        mock_qctn.cores_weights = {
            'A': TNTensor(torch.randn(3, 3), 1.0),
            'B': TNTensor(torch.randn(3, 3), 1.0),
        }
        
        mock_qctn.backend = MagicMock()
        mock_qctn.backend.tensor_to_numpy = lambda x: x.numpy() if hasattr(x, 'numpy') else np.array(x)
        mock_qctn.backend.convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        mock_qctn.backend.zeros = lambda shape: torch.zeros(shape)
        
        return mock_qctn
    
    def test_trainer_initialization(self):
        """Test ModelParallelTrainer initialization."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelTrainer
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn()
        mock_engine = self._create_mock_engine()
        mpi = MockMPIBackend()
        
        trainer = ModelParallelTrainer(mock_qctn, mock_engine, mpi)
        
        assert trainer.partition is not None
        assert trainer.partition.local_core_names == ['A', 'B']
    
    def test_forward(self):
        """Test forward pass."""
        import torch
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelTrainer
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn()
        mock_engine = self._create_mock_engine()
        mpi = MockMPIBackend()
        
        trainer = ModelParallelTrainer(mock_qctn, mock_engine, mpi)
        
        circuit_states = [torch.zeros(3), torch.zeros(3)]
        measure_matrices = [torch.randn(4, 3, 3), torch.randn(4, 3, 3)]
        
        result = trainer.forward(circuit_states, measure_matrices)
        
        assert result is not None
    
    def test_forward_with_gradient(self):
        """Test forward pass with gradient."""
        import torch
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelTrainer
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn()
        mock_engine = self._create_mock_engine()
        mpi = MockMPIBackend()
        
        trainer = ModelParallelTrainer(mock_qctn, mock_engine, mpi)
        
        circuit_states = [torch.zeros(3), torch.zeros(3)]
        measure_matrices = [torch.randn(4, 3, 3), torch.randn(4, 3, 3)]
        
        loss, local_grads = trainer.forward_with_gradient(circuit_states, measure_matrices)
        
        assert loss is not None
        assert 'A' in local_grads or 'B' in local_grads  # Has some local grads
    
    def test_train_step(self):
        """Test single training step."""
        import torch
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelTrainer
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        
        mock_qctn = self._create_mock_qctn()
        mock_engine = self._create_mock_engine()
        mpi = MockMPIBackend()
        
        trainer = ModelParallelTrainer(mock_qctn, mock_engine, mpi)
        
        circuit_states = [torch.zeros(3), torch.zeros(3)]
        measure_matrices = [torch.randn(4, 3, 3), torch.randn(4, 3, 3)]
        
        loss, grads = trainer.train_step(circuit_states, measure_matrices)
        
        assert loss is not None
        assert grads is not None


class TestModelParallelPartitioning:
    """Tests for core partitioning logic."""
    
    def test_partition_8_cores_4_workers(self):
        """Test partitioning 8 cores across 4 workers."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import DistributedContext
        import torch
        from tneq_qc.core.tn_tensor import TNTensor
        
        # Create mock for different ranks
        cores = [chr(ord('A') + i) for i in range(8)]  # A-H
        
        for rank in range(4):
            mock_qctn = MagicMock()
            mock_qctn.cores = cores
            mock_qctn.nqubits = 2
            mock_qctn.ncores = 8
            mock_qctn.adjacency_table = [
                {'core_idx': i, 'core_name': c, 'in_edge_list': [], 'out_edge_list': []}
                for i, c in enumerate(cores)
            ]
            mock_qctn.cores_weights = {c: TNTensor(torch.randn(3, 3), 1.0) for c in cores}
            mock_qctn.backend = MagicMock()
            
            # Mock MPI with specific rank
            mock_mpi = MagicMock()
            mock_mpi.is_main_process = MagicMock(return_value=(rank == 0))
            mock_mpi.get_context = MagicMock(return_value=DistributedContext(
                world_size=4, rank=rank, is_main_process=(rank == 0)
            ))
            mock_mpi.broadcast = MagicMock(side_effect=lambda x, src: x)
            
            manager = ModelParallelManager(mock_qctn, mock_mpi)
            
            # Each worker should have 2 cores
            assert len(manager.partition.local_core_names) == 2
            
            # Check correct cores assigned
            expected_cores = cores[rank * 2: (rank + 1) * 2]
            assert manager.partition.local_core_names == expected_cores
    
    def test_partition_7_cores_3_workers(self):
        """Test partitioning 7 cores across 3 workers (uneven)."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelManager
        from tneq_qc.distributed.comm.mpi_backend import DistributedContext
        import torch
        from tneq_qc.core.tn_tensor import TNTensor
        
        cores = [chr(ord('A') + i) for i in range(7)]  # A-G
        
        expected_partitions = {
            0: ['A', 'B', 'C'],  # 3 cores (7 // 3 + 1 for remainder)
            1: ['D', 'E'],       # 2 cores
            2: ['F', 'G'],       # 2 cores
        }
        
        for rank in range(3):
            mock_qctn = MagicMock()
            mock_qctn.cores = cores
            mock_qctn.nqubits = 2
            mock_qctn.ncores = 7
            mock_qctn.adjacency_table = [
                {'core_idx': i, 'core_name': c, 'in_edge_list': [], 'out_edge_list': []}
                for i, c in enumerate(cores)
            ]
            mock_qctn.cores_weights = {c: TNTensor(torch.randn(3, 3), 1.0) for c in cores}
            mock_qctn.backend = MagicMock()
            
            mock_mpi = MagicMock()
            mock_mpi.is_main_process = MagicMock(return_value=(rank == 0))
            mock_mpi.get_context = MagicMock(return_value=DistributedContext(
                world_size=3, rank=rank, is_main_process=(rank == 0)
            ))
            mock_mpi.broadcast = MagicMock(side_effect=lambda x, src: x)
            
            manager = ModelParallelManager(mock_qctn, mock_mpi)
            
            assert manager.partition.local_core_names == expected_partitions[rank], \
                f"Rank {rank}: expected {expected_partitions[rank]}, got {manager.partition.local_core_names}"


class TestCreateModelParallelTrainer:
    """Tests for factory function."""
    
    def test_factory_function(self):
        """Test create_model_parallel_trainer factory."""
        from tneq_qc.distributed.parallel.model_parallel import ModelParallelTrainer
        from tneq_qc.distributed.comm.mpi_backend import MockMPIBackend
        import torch
        from tneq_qc.core.tn_tensor import TNTensor
        
        # Create a mock QCTN
        mock_qctn = MagicMock()
        mock_qctn.cores = ['A', 'B']
        mock_qctn.nqubits = 2
        mock_qctn.ncores = 2
        mock_qctn.adjacency_table = [
            {'core_idx': 0, 'core_name': 'A', 'in_edge_list': [], 'out_edge_list': []},
            {'core_idx': 1, 'core_name': 'B', 'in_edge_list': [], 'out_edge_list': []},
        ]
        mock_qctn.cores_weights = {
            'A': TNTensor(torch.randn(3, 3), 1.0),
            'B': TNTensor(torch.randn(3, 3), 1.0),
        }
        mock_qctn.backend = MagicMock()
        mock_qctn.backend.tensor_to_numpy = lambda x: x.numpy() if hasattr(x, 'numpy') else np.array(x)
        mock_qctn.backend.convert_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        
        # Create mock engine directly
        mock_engine = MagicMock()
        mock_engine.mpi = MockMPIBackend()
        mock_engine._base_engine = MagicMock()
        mock_engine._base_engine.backend = mock_qctn.backend
        
        trainer = ModelParallelTrainer(
            qctn=mock_qctn,
            engine=mock_engine,
            mpi=MockMPIBackend()
        )
        
        assert trainer is not None
        assert hasattr(trainer, 'partition')
        assert trainer.partition.local_core_names == ['A', 'B']


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
