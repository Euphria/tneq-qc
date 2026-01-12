"""
Parallel training strategies module.
"""

from .data_parallel import DataParallelTrainer, TrainingConfig, TrainingStats
from .model_parallel import (
    ModelParallelManager,
    ModelParallelTrainer,
    ModelParallelConfig,
    CorePartition,
    create_model_parallel_trainer
)

__all__ = [
    # Data Parallel
    'DataParallelTrainer', 
    'TrainingConfig',
    'TrainingStats',
    # Model Parallel
    'ModelParallelManager',
    'ModelParallelTrainer',
    'ModelParallelConfig',
    'CorePartition',
    'create_model_parallel_trainer',
]
