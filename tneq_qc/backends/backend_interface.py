"""
Backend interface for computational backends.

This module defines the abstract base class and information structure
for computational backends.
"""

from __future__ import annotations
from typing import Optional, Any
from abc import ABC, abstractmethod
import numpy as np


class BackendInfo:
    """
    Simple information class for backend configuration.
    
    This class only records backend information, without providing
    tensor allocation or conversion methods. The actual tensor operations
    should be implemented in the caller based on this information.
    """
    
    def __init__(self, backend_type: str, device: Optional[str] = None, **kwargs):
        """
        Initialize backend information.
        
        Args:
            backend_type (str): Type of backend ('jax', 'pytorch', etc.).
            device (Optional[str]): Device specification (e.g., 'cpu', 'cuda', 'cuda:0', 'gpu').
            **kwargs: Additional backend-specific configuration.
        """
        self.backend_type = backend_type.lower()
        self.device = device
        self.config = kwargs
    
    def __repr__(self):
        return f"BackendInfo(backend_type='{self.backend_type}', device='{self.device}', config={self.config})"
    
    def __str__(self):
        return self.__repr__()

class ComputeBackend(ABC):
    """
    Abstract base class for computational backends.
    
    All backends must implement these methods for tensor operations,
    gradient computation, and JIT compilation.
    """
    
    def __init__(self):
        """Initialize backend with BackendInfo."""
        self.backend_info: Optional[BackendInfo] = None

    @abstractmethod
    def execute_expression(self, expression, *tensors):
        """
        Execute a contraction expression with given tensors.
        
        Args:
            expression: Optimized contraction expression (opt_einsum.ContractExpression).
            *tensors: Variable number of input tensors.
        
        Returns:
            Tensor result of the contraction.
        """
        pass

    @abstractmethod
    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Create a function that computes both value and gradient.
        
        Args:
            loss_fn: Loss function to differentiate.
            argnums: Argument indices to compute gradients for.
        
        Returns:
            Function that returns (value, gradients).
        """
        pass

    @abstractmethod
    def jit_compile(self, func):
        """
        JIT compile a function for faster execution.
        
        Args:
            func: Function to compile.
        
        Returns:
            JIT-compiled function.
        """
        pass

    @abstractmethod
    def convert_to_tensor(self, array):
        """
        Convert array-like object to backend tensor.
        
        Args:
            array: Array-like object (numpy array, list, etc.).
        
        Returns:
            Backend-specific tensor.
        """
        pass

    @abstractmethod
    def optimizer_update(self, params, grads, state, method: str, hyperparams: dict):
        """
        Perform a single optimization step.
        
        Args:
            params: Current parameters (Dict[key, Tensor] or List[Tensor])
            grads: Corresponding gradients
            state: Optimizer state (e.g. momentum, Adam's m/v), structure maintained by Backend
            method: Optimization method name ('adam', 'sgdg', etc.)
            hyperparams: Hyperparameters (lr, beta1, beta2, momentum, etc.)
            
        Returns:
            (new_params, new_state): Updated parameters and state
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        pass
    
    def get_backend_info(self) -> BackendInfo:
        """
        Get the BackendInfo instance for this backend.
        
        Returns:
            BackendInfo instance.
        """
        if self.backend_info is None:
            # Create default BackendInfo
            self.backend_info = BackendInfo(self.get_backend_name())
        return self.backend_info
    
    def set_backend_info(self, backend_info: BackendInfo):
        """
        Set the BackendInfo instance for this backend.
        
        Args:
            backend_info (BackendInfo): BackendInfo instance to use.
        """
        if backend_info.backend_type != self.get_backend_name():
            raise ValueError(
                f"BackendInfo type '{backend_info.backend_type}' does not match "
                f"backend '{self.get_backend_name()}'"
            )
        self.backend_info = backend_info

    @abstractmethod
    def init_random_core(self, shape):
        """
        Initialize a random core tensor (orthogonal initialization).
        
        Args:
            shape: Shape of the tensor.
            
        Returns:
            Initialized tensor.
        """
        pass

    @abstractmethod
    def get_tensor_type(self):
        """
        Get the type of tensors used by this backend.
        
        Returns:
            Type/Class of the tensor.
        """
        pass

    @abstractmethod
    def tensor_to_numpy(self, tensor) -> np.ndarray:
        """
        Convert a backend tensor to a NumPy array.

        Args:
            tensor: Backend-specific tensor.

        Returns:
            NumPy ndarray with the same data.
        """
        pass

    @abstractmethod
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Integer seed value.
        """
        pass
