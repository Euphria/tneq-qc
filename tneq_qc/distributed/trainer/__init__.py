"""
Distributed Trainer module.

Provides high-level distributed training API with:
- DistributedConfig: Configuration for distributed training
- DistributedTrainer: Main trainer class
"""

from .distributed_trainer import DistributedTrainer, DistributedConfig

__all__ = ['DistributedTrainer', 'DistributedConfig']
