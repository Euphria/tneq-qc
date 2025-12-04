"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch

from ..contractor import EinsumStrategy, StrategyCompiler
from ..backends.backend_factory import BackendFactory, ComputeBackend


class EngineSiamese:
    """
    EngineSiamese that combines tensor contraction expression generation with backend execution.
    
    This class separates concerns:
    - EinsumStrategy: Generates einsum expressions using opt_einsum (legacy)
    - StrategyCompiler: Compiles optimal strategies based on network structure
    - ComputeBackend: Executes expressions using JAX, PyTorch, etc.
    """

    def __init__(self, backend: Optional[Union[str, ComputeBackend]] = None, strategy_mode: str = 'balanced'):
        """
        Initialize the engine with a specific backend and strategy mode.
        
        Args:
            backend (str or ComputeBackend, optional): Backend to use. 
                Can be 'jax', 'pytorch', or a ComputeBackend instance.
                If None, uses the default backend.
            strategy_mode (str): Contraction strategy mode:
                - 'fast': Use einsum only (fastest compilation)
                - 'balanced': Use einsum + MPS chain (balanced)
                - 'full': Use all available strategies (slowest compilation, best runtime)
        """
        if backend is None:
            self.backend = BackendFactory.get_default_backend()
        elif isinstance(backend, str):
            self.backend = BackendFactory.create_backend(backend, device="cuda")
        else:
            self.backend = backend

        self.contractor = EinsumStrategy()  # Keep for legacy methods
        self.strategy_compiler = StrategyCompiler(mode=strategy_mode)
        self.strategy_mode = strategy_mode

    # ============================================================================
    # Strategy-based Compilation Methods (NEW API)
    # ============================================================================

    def contract_with_compiled_strategy(self, qctn, circuit_states=None, measure_input=None, 
                                       measure_is_matrix=True, force_recompile: bool = False):
        """
        Contract using compiled strategy (auto-selected based on mode).
        
        This is the NEW recommended API that automatically selects the best strategy.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states.
            measure_input (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
            force_recompile (bool): Force recompilation even if cached.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Prepare shapes_info
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if force_recompile or not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            print(f"[EngineSiamese] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            print(f"[EngineSiamese] Using cached strategy: {strategy_name}")
        
        # Prepare data
        cores_dict = {name: self.backend.convert_to_tensor(qctn.cores_weights[name]) for name in qctn.cores}
        
        # Execute
        result = compute_fn(cores_dict, circuit_states, measure_input)
        
        return result

    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list=None, measure_input_list=None, 
                                                    measure_is_matrix=True, force_recompile: bool = False) -> Tuple:
        """
        Contract using compiled strategy and compute gradients.
        
        This is the NEW recommended API for gradient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (array or list, optional): Circuit input states.
            measure_input_list (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
            force_recompile (bool): Force recompilation even if cached.
        
        Returns:
            tuple: (loss, gradients)
        """
        circuit_states = circuit_states_list
        measure_input = measure_input_list

        # Prepare shapes_info
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if force_recompile or not hasattr(qctn, cache_key):
            # Compile strategy
            compute_fn, strategy_name, cost = self.strategy_compiler.compile(qctn, shapes_info, self.backend)
            
            # Cache the result
            setattr(qctn, cache_key, {
                'compute_fn': compute_fn,
                'strategy_name': strategy_name,
                'cost': cost
            })
            print(f"[EngineSiamese] Compiled and cached strategy: {strategy_name}")
        else:
            cached = getattr(qctn, cache_key)
            compute_fn = cached['compute_fn']
            strategy_name = cached['strategy_name']
            # print(f"[EngineSiamese] Using cached strategy: {strategy_name}")
            
        # Define loss function
        def loss_fn(*core_tensors):
            # Reconstruct cores_dict
            cores_dict = {name: tensor for name, tensor in zip(qctn.cores, core_tensors)}
            
            # Execute contraction
            result = compute_fn(cores_dict, circuit_states, measure_input)
            
            # Compute Cross Entropy loss
            # Target is all ones (maximizing probability)
            target = torch.ones_like(result)
            
            # Avoid log(0)
            result = torch.clamp(result, min=1e-10)
            log_result = torch.log(result)
            return -torch.mean(target * log_result)
        
        # Prepare core tensors
        core_tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Compute gradients
        # We want gradients with respect to all cores
        argnums = list(range(len(core_tensors)))
        
        # Create value_and_grad function
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        
        # Execute
        loss, grads = value_and_grad_fn(*core_tensors)
        
        return loss, grads
