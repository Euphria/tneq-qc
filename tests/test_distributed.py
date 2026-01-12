"""
Tests for Distributed Training Module

Tests the MPI backend, data parallel trainer, and distributed engine.
These tests use MockCommMPI to run without actual MPI.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCommMPI:
    """Tests for MPI communication backend."""
    
    def test_mock_backend_initialization(self):
        """Test MockCommMPI initialization."""
        from tneq_qc.distributed.comm import MockCommMPI, DistributedContext
        
        backend = MockCommMPI()
        
        assert backend.rank == 0
        assert backend.world_size == 1
        assert backend.is_main_process() == True
        
        ctx = backend.get_context()
        assert isinstance(ctx, DistributedContext)
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.is_main_process == True
    
    def test_mock_backend_broadcast(self):
        """Test broadcast operation with mock backend."""
        import numpy as np
        from tneq_qc.distributed.comm import MockCommMPI
        
        backend = MockCommMPI()
        
        arr = np.random.randn(3, 4)
        result = backend.broadcast(arr, src=0)
        
        assert result.shape == arr.shape
        assert np.allclose(result, arr)
    
    def test_mock_backend_allreduce(self):
        """Test allreduce operation with mock backend."""
        import numpy as np
        from tneq_qc.distributed.comm import MockCommMPI, ReduceOp
        
        backend = MockCommMPI()
        
        arr = np.random.randn(3, 4)
        result = backend.allreduce(arr, op=ReduceOp.SUM)
        
        assert result.shape == arr.shape
        assert np.allclose(result, arr)
    
    def test_mock_backend_allgather(self):
        """Test allgather operation with mock backend."""
        import numpy as np
        from tneq_qc.distributed.comm import MockCommMPI
        
        backend = MockCommMPI()
        
        arr = np.random.randn(3, 4)
        result = backend.allgather(arr)
        
        assert len(result) == 1
        assert np.allclose(result[0], arr)
    
    def test_mock_backend_allreduce_list(self):
        """Test allreduce on array list with mock backend."""
        import numpy as np
        from tneq_qc.distributed.comm import MockCommMPI, ReduceOp
        
        backend = MockCommMPI()
        
        arrays = [np.random.randn(3, 4), np.random.randn(5, 6)]
        results = backend.allreduce_list(arrays, op=ReduceOp.AVG)
        
        assert len(results) == len(arrays)
        for r, a in zip(results, arrays):
            assert r.shape == a.shape
            assert np.allclose(r, a)
    
    def test_mock_backend_allreduce_scalar(self):
        """Test scalar allreduce with mock backend."""
        from tneq_qc.distributed.comm import MockCommMPI, ReduceOp
        
        backend = MockCommMPI()
        
        value = 3.14
        result = backend.allreduce_scalar(value, op=ReduceOp.SUM)
        
        assert result == value
    
    def test_get_backend_mock(self):
        """Test get_comm_mpi with use_mpi=False."""
        from tneq_qc.distributed.comm import get_comm_mpi, MockCommMPI
        
        backend = get_comm_mpi(use_mpi=False)
        assert isinstance(backend, MockCommMPI)
    
    def test_reduce_op_values(self):
        """Test ReduceOp enum values."""
        from tneq_qc.distributed.comm import ReduceOp
        
        assert ReduceOp.SUM.value == "SUM"
        assert ReduceOp.AVG.value == "AVG"
        assert ReduceOp.MAX.value == "MAX"
        assert ReduceOp.MIN.value == "MIN"


class TestDistributedContext:
    """Tests for DistributedContext."""
    
    def test_context_repr(self):
        """Test context string representation."""
        from tneq_qc.distributed.comm import DistributedContext
        
        ctx = DistributedContext(
            world_size=4,
            rank=2,
            local_rank=0,
            is_main_process=False
        )
        
        repr_str = repr(ctx)
        assert "rank=2/4" in repr_str
        assert "main=False" in repr_str
    
    def test_context_main_process(self):
        """Test main process detection."""
        from tneq_qc.distributed.comm import DistributedContext
        
        main_ctx = DistributedContext(
            world_size=4, rank=0, local_rank=0, is_main_process=True
        )
        worker_ctx = DistributedContext(
            world_size=4, rank=1, local_rank=0, is_main_process=False
        )
        
        assert main_ctx.is_main_process == True
        assert worker_ctx.is_main_process == False


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_config(self):
        """Test default training configuration."""
        from tneq_qc.distributed.parallel import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.max_steps == 1000
        assert config.log_interval == 10
        assert config.learning_rate == 1e-2
        assert config.optimizer_method == 'sgdg'
        assert config.momentum == 0.9
        assert config.stiefel == True
    
    def test_custom_config(self):
        """Test custom training configuration."""
        from tneq_qc.distributed.parallel import TrainingConfig
        
        config = TrainingConfig(
            max_steps=500,
            learning_rate=0.001,
            optimizer_method='adam',
            lr_schedule=[(0, 0.01), (100, 0.001)]
        )
        
        assert config.max_steps == 500
        assert config.learning_rate == 0.001
        assert config.optimizer_method == 'adam'
        assert len(config.lr_schedule) == 2


class TestTrainingStats:
    """Tests for TrainingStats."""
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        from tneq_qc.distributed.parallel.data_parallel import TrainingStats
        
        stats = TrainingStats()
        
        assert stats.final_loss == float('inf')
        assert stats.total_steps == 0
        assert stats.total_time == 0.0
        assert stats.losses == []
        assert stats.converged == False
    
    def test_stats_to_dict(self):
        """Test stats to dictionary conversion."""
        from tneq_qc.distributed.parallel.data_parallel import TrainingStats
        
        stats = TrainingStats(
            final_loss=0.5,
            total_steps=100,
            total_time=10.0,
            converged=True
        )
        
        d = stats.to_dict()
        
        assert d['final_loss'] == 0.5
        assert d['total_steps'] == 100
        assert d['total_time'] == 10.0
        assert d['converged'] == True


class TestDataParallelTrainer:
    """Tests for DataParallelTrainer."""
    
    @pytest.fixture
    def setup_trainer(self):
        """Setup trainer with mock components."""
        import torch
        from tneq_qc.distributed.comm import MockCommMPI
        from tneq_qc.distributed.parallel import DataParallelTrainer, TrainingConfig
        from tneq_qc.core.engine_siamese import EngineSiamese
        from tneq_qc.core.qctn import QCTN, QCTNHelper
        
        # Create simple QCTN
        graph = QCTNHelper.generate_example_graph(n=4)
        backend = "pytorch"
        
        engine = EngineSiamese(backend=backend, strategy_mode='fast')
        qctn = QCTN(graph, backend=engine.backend)
        config = TrainingConfig(max_steps=10, log_interval=5)
        mpi = MockCommMPI()
        
        trainer = DataParallelTrainer(
            engine=engine,
            qctn=qctn,
            config=config,
            mpi_backend=mpi
        )
        
        return trainer, engine, qctn
    
    def test_trainer_initialization(self, setup_trainer):
        """Test trainer initialization."""
        trainer, engine, qctn = setup_trainer
        
        assert trainer.engine is engine
        assert trainer.qctn is qctn
        assert trainer.global_step == 0
        assert trainer.mpi.is_main_process()
    
    def test_partition_data(self, setup_trainer):
        """Test data partitioning."""
        trainer, _, _ = setup_trainer
        
        data_list = [{'id': i} for i in range(10)]
        local_data = trainer.partition_data(data_list)
        
        # With world_size=1, should get all data
        assert len(local_data) == 10
    
    def test_sync_gradients(self, setup_trainer):
        """Test gradient synchronization."""
        import torch
        
        trainer, _, _ = setup_trainer
        
        grads = [torch.randn(3, 3), torch.randn(4, 4)]
        synced = trainer.sync_gradients(grads)
        
        assert len(synced) == len(grads)
        for s, g in zip(synced, grads):
            assert s.shape == g.shape
    
    def test_sync_loss(self, setup_trainer):
        """Test loss synchronization."""
        trainer, _, _ = setup_trainer
        
        loss = 0.5
        synced_loss = trainer.sync_loss(loss)
        
        assert synced_loss == loss
    
    def test_accumulate_gradients(self, setup_trainer):
        """Test gradient accumulation."""
        import torch
        
        trainer, _, _ = setup_trainer
        
        grads1 = [torch.ones(3, 3), torch.ones(4, 4)]
        grads2 = [torch.ones(3, 3) * 2, torch.ones(4, 4) * 2]
        
        trainer.accumulate_gradients(grads1)
        trainer.accumulate_gradients(grads2)
        
        avg_grads = trainer.get_accumulated_gradients()
        
        assert len(avg_grads) == 2
        assert torch.allclose(avg_grads[0], torch.ones(3, 3) * 1.5)
        assert torch.allclose(avg_grads[1], torch.ones(4, 4) * 1.5)


class TestDistributedEngineSiamese:
    """Tests for DistributedEngineSiamese."""
    
    def test_engine_initialization(self):
        """Test distributed engine initialization."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        from tneq_qc.distributed.comm import MockCommMPI
        
        mpi = MockCommMPI()
        engine = DistributedEngineSiamese(
            backend='pytorch',
            strategy_mode='fast',
            mpi_backend=mpi
        )
        
        assert engine.backend is not None
        assert engine.mpi is mpi
        assert engine.enable_tensor_parallel == False
    
    def test_engine_backend_proxy(self):
        """Test that engine proxies to base engine."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        from tneq_qc.distributed.comm import MockCommMPI
        
        mpi = MockCommMPI()
        engine = DistributedEngineSiamese(
            backend='pytorch',
            mpi_backend=mpi
        )
        
        # Check proxy properties work
        assert engine.backend is engine._base_engine.backend
        assert engine.contractor is engine._base_engine.contractor
        assert engine.strategy_compiler is engine._base_engine.strategy_compiler
    
    def test_partition_measure_matrices(self):
        """Test measurement matrix partitioning."""
        import torch
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        from tneq_qc.distributed.comm import MockCommMPI
        
        mpi = MockCommMPI()
        engine = DistributedEngineSiamese(backend='pytorch', mpi_backend=mpi)
        
        # Create mock measure matrices
        B, K = 16, 3
        measure_list = [torch.randn(B, K, K) for _ in range(2)]
        
        partitioned = engine.partition_measure_matrices(measure_list)
        
        # With world_size=1, should get all data
        assert len(partitioned) == 2
        assert partitioned[0].shape[0] == B
    
    def test_tensor_parallel_not_enabled(self):
        """Test tensor parallel raises error when not enabled."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        from tneq_qc.distributed.comm import MockCommMPI
        from tneq_qc.core.qctn import QCTN
        
        mpi = MockCommMPI()
        engine = DistributedEngineSiamese(backend='pytorch', mpi_backend=mpi)
        
        qctn = QCTN("-3-A-3-", backend=engine.backend)
        
        with pytest.raises(RuntimeError):
            engine.contract_tensor_parallel(qctn, [], [])
    
    def test_setup_tensor_parallel(self):
        """Test tensor parallel configuration."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        from tneq_qc.distributed.comm import MockCommMPI
        
        mpi = MockCommMPI()
        engine = DistributedEngineSiamese(backend='pytorch', mpi_backend=mpi)
        
        config = {'partition_dim': 'batch', 'min_size': 100}
        engine.setup_tensor_parallel(config)
        
        assert engine.enable_tensor_parallel == True
        assert engine.tensor_parallel_config == config


class TestDistributedTrainer:
    """Tests for DistributedTrainer high-level API."""
    
    def test_trainer_initialization(self):
        """Test distributed trainer initialization."""
        from tneq_qc.distributed.trainer import DistributedTrainer
        
        config = {
            'backend_type': 'pytorch',
            'qctn_graph': '-3-A-3-B-3-',
            'max_steps': 10,
        }
        
        # Note: This will try to use real MPI, falling back to mock
        try:
            trainer = DistributedTrainer(config)
            assert trainer.qctn is not None
            assert trainer.engine is not None
        except Exception as e:
            # If MPI is not available, this is expected
            if "mpi4py" not in str(e).lower():
                raise


class TestIntegration:
    """Integration tests for distributed training."""
    
    def test_simple_training_mock(self):
        """Test simple training flow with mock MPI."""
        import torch
        from tneq_qc.distributed.comm import MockCommMPI
        from tneq_qc.distributed.parallel import DataParallelTrainer, TrainingConfig
        from tneq_qc.core.engine_siamese import EngineSiamese
        from tneq_qc.core.qctn import QCTN
        
        # Setup - use 'balanced' mode which works with greedy strategy
        graph = "-3-A-3-B-3-"
        engine = EngineSiamese(backend='pytorch', strategy_mode='balanced')
        qctn = QCTN(graph, backend=engine.backend)
        mpi = MockCommMPI()
        
        config = TrainingConfig(
            max_steps=5,
            log_interval=2,
            learning_rate=0.01
        )
        
        trainer = DataParallelTrainer(
            engine=engine,
            qctn=qctn,
            config=config,
            mpi_backend=mpi
        )
        
        # Generate simple data
        D = qctn.nqubits
        K = 3
        B = 4
        
        x = torch.randn(B, D)
        Mx_list, _ = engine.generate_data(x, K=K)
        
        data_list = [{"measure_input_list": Mx_list}]
        circuit_states = [torch.zeros(K) for _ in range(D)]
        for s in circuit_states:
            s[-1] = 1.0
        
        # Train
        stats = trainer.train(data_list, circuit_states)
        
        assert stats.total_steps == 5
        assert len(stats.losses) == 5
        assert stats.final_loss < float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
