from ..config import Configuration
from .qctn import QCTN, QCTNHelper
import jax
import jax.numpy as jnp
import itertools
from functools import reduce

class ContractorQCTN:
    """
    ContractorQCTN class for contracting quantum circuit tensor networks.
    
    This class provides methods to contract quantum circuit tensor networks using JAX.
    It supports both contraction with inputs and contraction with another QCTN instance.
    """

    @staticmethod
    def contract(qctn, inputs=None):
        """
        Contract the quantum circuit tensor network with given inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (jnp.ndarray or dict, optional): The inputs for the contraction operation.
        
        Returns:
            jnp.ndarray: The result of the contraction operation.
        """
        return qctn.contract(attach=inputs)

    @staticmethod
    def contract_with_QCTN_for_core_gradient(qctn, inputs=None):
        """
        Contract the quantum circuit tensor network with another QCTN instance for core gradient.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (jnp.ndarray or dict, optional): The inputs for the contraction operation.
        
        Returns:
            jnp.ndarray: The result of the contraction operation.
        """
        return qctn.contract_with_QCTN_for_core_gradient(attach=inputs)