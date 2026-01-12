"""
Distributed Engine module.

Provides DistributedEngineSiamese with:
- QCTN graph partitioning across workers
- Hierarchical tensor contraction (log(n)+1 stages)
- Tensor parallel matrix multiplication
"""

from .distributed_engine import (
    DistributedEngineSiamese,
    PartitionConfig,
    ContractStage,
    DistributedContractPlan,
)

__all__ = [
    'DistributedEngineSiamese',
    'PartitionConfig',
    'ContractStage',
    'DistributedContractPlan',
]
