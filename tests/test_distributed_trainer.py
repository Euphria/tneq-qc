"""
Tests for DistributedTrainer Initialization

Tests the DistributedTrainer and DistributedConfig classes.
These tests use mock communication backends to run without actual MPI.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from tneq_qc.distributed import DistributedConfig
        
        config = DistributedConfig()
        
        # Backend configuration
        assert config.backend_type == 'pytorch'
        assert config.device == 'cpu'
        assert config.strategy_mode == 'balanced'
        assert config.mx_K == 100
        
        # QCTN configuration
        assert config.qctn_graph is None
        assert config.num_qubits == 4
        
        # Communication configuration
        assert config.comm_backend == 'auto'
        assert config.use_distributed == True
        
        # Partitioning configuration
        assert config.partition_strategy == 'layer'
        assert config.min_cores_per_partition == 1
        assert config.balance_partitions == True
        
        # Training configuration
        assert config.max_steps == 1000
        assert config.log_interval == 10
        assert config.learning_rate == 1e-2
        assert config.optimizer == 'sgdg'
        assert config.momentum == 0.9
        assert config.stiefel == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from tneq_qc.distributed import DistributedConfig
        
        config = DistributedConfig(
            backend_type='pytorch',
            qctn_graph='-3-A-3-B-3-',
            num_qubits=2,
            max_steps=500,
            learning_rate=0.001,
            partition_strategy='core',
            use_distributed=False,
        )
        
        assert config.qctn_graph == '-3-A-3-B-3-'
        assert config.num_qubits == 2
        assert config.max_steps == 500
        assert config.learning_rate == 0.001
        assert config.partition_strategy == 'core'
        assert config.use_distributed == False
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        from tneq_qc.distributed import DistributedConfig
        
        config_dict = {
            'backend_type': 'pytorch',
            'max_steps': 200,
            'learning_rate': 0.005,
            'num_qubits': 3,
            'unknown_key': 'should_be_ignored',
        }
        
        config = DistributedConfig.from_dict(config_dict)
        
        assert config.backend_type == 'pytorch'
        assert config.max_steps == 200
        assert config.learning_rate == 0.005
        assert config.num_qubits == 3
        # Unknown keys should be ignored (no error)
    
    def test_to_training_config(self):
        """Test conversion to TrainingConfig."""
        from tneq_qc.distributed import DistributedConfig
        from tneq_qc.distributed.parallel import TrainingConfig
        
        config = DistributedConfig(
            max_steps=100,
            log_interval=5,
            learning_rate=0.01,
            optimizer='sgdg',
            momentum=0.95,
            stiefel=True,
        )
        
        training_config = config.to_training_config()
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.max_steps == 100
        assert training_config.log_interval == 5
        assert training_config.learning_rate == 0.01
        assert training_config.optimizer_method == 'sgdg'
        assert training_config.momentum == 0.95
        assert training_config.stiefel == True
    
    def test_to_partition_config(self):
        """Test conversion to PartitionConfig."""
        from tneq_qc.distributed import DistributedConfig, PartitionConfig
        
        config = DistributedConfig(
            partition_strategy='layer',
            min_cores_per_partition=2,
            balance_partitions=True,
        )
        
        partition_config = config.to_partition_config(world_size=4)
        
        assert isinstance(partition_config, PartitionConfig)
        assert partition_config.strategy == 'layer'
        assert partition_config.num_partitions == 4
        assert partition_config.min_cores_per_partition == 2
        assert partition_config.balance_partitions == True


class TestDistributedTrainerInitialization:
    """Tests for DistributedTrainer initialization."""
    
    def test_init_with_config_object(self):
        """Test initialization with DistributedConfig object."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        
        config = DistributedConfig(
            backend_type='pytorch',
            qctn_graph='-3-A-3-B-3-',
            use_distributed=False,  # Use mock comm
            max_steps=10,
        )
        
        trainer = DistributedTrainer(config)
        
        # Check trainer attributes
        assert trainer.config == config
        assert trainer.comm is not None
        assert trainer.comm.rank == 0
        assert trainer.comm.world_size == 1
        assert trainer.engine is not None
        assert trainer.qctn is not None
    
    def test_init_with_dict(self):
        """Test initialization with dictionary config."""
        from tneq_qc.distributed import DistributedTrainer
        
        config_dict = {
            'backend_type': 'pytorch',
            'qctn_graph': '-3-A-3-B-3-',
            'use_distributed': False,
            'max_steps': 10,
        }
        
        trainer = DistributedTrainer(config_dict)
        
        # Check config was parsed correctly
        assert trainer.config.backend_type == 'pytorch'
        assert trainer.config.qctn_graph == '-3-A-3-B-3-'
        assert trainer._raw_config == config_dict
    
    def test_init_creates_qctn(self):
        """Test that initialization creates QCTN model."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        from tneq_qc.core.qctn import QCTN
        
        config = DistributedConfig(
            qctn_graph='-3-A-3-B-3-C-3-',
            use_distributed=False,
        )
        
        trainer = DistributedTrainer(config)
        
        assert trainer.qctn is not None
        assert isinstance(trainer.qctn, QCTN)
        assert trainer.qctn.nqubits == 1  # Single line graph = 1 qubit
        assert len(trainer.qctn.cores) == 3  # A, B, C
    
    def test_init_creates_default_qctn(self):
        """Test that initialization creates default QCTN when graph is None."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        
        config = DistributedConfig(
            qctn_graph=None,
            num_qubits=4,
            use_distributed=False,
        )
        
        trainer = DistributedTrainer(config)
        
        assert trainer.qctn is not None
        assert trainer.qctn.nqubits == 4
    
    def test_init_creates_engine(self):
        """Test that initialization creates DistributedEngineSiamese."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        from tneq_qc.distributed.engine import DistributedEngineSiamese
        
        config = DistributedConfig(
            strategy_mode='fast',
            use_distributed=False,
        )
        
        trainer = DistributedTrainer(config)
        
        assert trainer.engine is not None
        assert isinstance(trainer.engine, DistributedEngineSiamese)
        assert trainer.engine.backend is not None
    
    def test_init_creates_checkpoint_dir(self, tmp_path):
        """Test that initialization creates checkpoint directory."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        
        checkpoint_dir = tmp_path / "test_checkpoints"
        
        config = DistributedConfig(
            use_distributed=False,
            checkpoint_dir=str(checkpoint_dir),
        )
        
        trainer = DistributedTrainer(config)
        
        assert checkpoint_dir.exists()
        assert trainer.checkpoint_dir == checkpoint_dir
    
    def test_init_comm_backend_auto(self):
        """Test auto communication backend selection."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        from tneq_qc.distributed.comm import MockCommMPI
        
        config = DistributedConfig(
            comm_backend='auto',
            use_distributed=False,
        )
        
        trainer = DistributedTrainer(config)
        
        # With use_distributed=False, should get mock backend
        assert isinstance(trainer.comm, MockCommMPI)
    
    def test_init_comm_backend_mpi(self):
        """Test MPI communication backend selection."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        from tneq_qc.distributed.comm import MockCommMPI
        
        config = DistributedConfig(
            comm_backend='mpi',
            use_distributed=False,
        )
        
        trainer = DistributedTrainer(config)
        
        # With use_distributed=False, should get mock backend
        assert isinstance(trainer.comm, MockCommMPI)


class TestDistributedTrainerProperties:
    """Tests for DistributedTrainer properties and backward compatibility."""
    
    def test_mpi_property(self):
        """Test mpi property for backward compatibility."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        
        config = DistributedConfig(use_distributed=False)
        trainer = DistributedTrainer(config)
        
        # mpi should alias comm
        assert trainer.mpi is trainer.comm
    
    def test_ctx_property(self):
        """Test ctx property returns DistributedContext."""
        from tneq_qc.distributed import DistributedTrainer, DistributedConfig
        from tneq_qc.distributed.comm import DistributedContext
        
        config = DistributedConfig(use_distributed=False)
        trainer = DistributedTrainer(config)
        
        ctx = trainer.ctx
        
        assert isinstance(ctx, DistributedContext)
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.is_main_process == True


class TestDistributedEngineInitialization:
    """Tests for DistributedEngineSiamese initialization."""
    
    def test_engine_init_with_mock_comm(self):
        """Test engine initialization with mock communication."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese, PartitionConfig
        from tneq_qc.distributed.comm import MockCommMPI
        
        comm = MockCommMPI()
        partition_config = PartitionConfig(num_partitions=1)
        
        engine = DistributedEngineSiamese(
            backend='pytorch',
            strategy_mode='fast',
            comm=comm,
            partition_config=partition_config,
        )
        
        assert engine.comm is comm
        assert engine.rank == 0
        assert engine.world_size == 1
        assert engine.backend is not None
    
    def test_engine_init_distributed(self):
        """Test engine.init_distributed() partitions QCTN."""
        from tneq_qc.distributed.engine import DistributedEngineSiamese, PartitionConfig
        from tneq_qc.distributed.comm import MockCommMPI
        from tneq_qc.core.qctn import QCTN
        
        comm = MockCommMPI()
        partition_config = PartitionConfig(num_partitions=1)
        
        engine = DistributedEngineSiamese(
            backend='pytorch',
            comm=comm,
            partition_config=partition_config,
        )
        
        # Create QCTN
        qctn = QCTN('-3-A-3-B-3-', backend=engine.backend)
        
        # Initialize distributed contraction
        plan = engine.init_distributed(qctn)
        
        assert plan is not None
        assert engine._is_initialized == True
        assert engine._qctn is qctn
        assert len(plan.local_cores) > 0


class TestPartitionConfig:
    """Tests for PartitionConfig dataclass."""
    
    def test_default_values(self):
        """Test default partition configuration."""
        from tneq_qc.distributed.engine import PartitionConfig
        
        config = PartitionConfig()
        
        assert config.strategy == 'layer'
        assert config.num_partitions == 1
        assert config.min_cores_per_partition == 1
        assert config.balance_partitions == True
    
    def test_custom_values(self):
        """Test custom partition configuration."""
        from tneq_qc.distributed.engine import PartitionConfig
        
        config = PartitionConfig(
            strategy='core',
            num_partitions=4,
            min_cores_per_partition=2,
            balance_partitions=False,
        )
        
        assert config.strategy == 'core'
        assert config.num_partitions == 4
        assert config.min_cores_per_partition == 2
        assert config.balance_partitions == False


class TestContractPlan:
    """Tests for DistributedContractPlan."""
    
    def test_contract_plan_creation(self):
        """Test contract plan is created correctly."""
        from tneq_qc.distributed.engine import (
            DistributedEngineSiamese, 
            PartitionConfig,
            DistributedContractPlan,
        )
        from tneq_qc.distributed.comm import MockCommMPI
        from tneq_qc.core.qctn import QCTN
        
        comm = MockCommMPI()
        engine = DistributedEngineSiamese(
            backend='pytorch',
            comm=comm,
            partition_config=PartitionConfig(num_partitions=1),
        )
        
        qctn = QCTN('-3-A-3-B-3-C-3-', backend=engine.backend)
        plan = engine.init_distributed(qctn)
        
        assert isinstance(plan, DistributedContractPlan)
        assert plan.num_stages >= 1
        assert len(plan.stages) == plan.num_stages
        assert plan.local_partition_idx == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
