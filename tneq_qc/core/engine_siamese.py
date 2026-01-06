"""
Unified engine that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use EinsumStrategy to generate
expressions and then execute them using the specified backend.

Now supports strategy-based compilation for optimized contraction paths.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

from ..contractor import EinsumStrategy, StrategyCompiler, GreedyStrategy
from ..backends.backend_factory import BackendFactory, ComputeBackend
from .tn_tensor import TNTensor


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
            self.backend = BackendFactory.create_backend(backend, device="cpu")
        else:
            self.backend = backend

        self.contractor = EinsumStrategy()  # Keep for legacy methods
        self.strategy_compiler = StrategyCompiler(mode=strategy_mode)
        self.strategy_mode = strategy_mode

    # ============================================================================
    # Strategy-based Compilation Methods (NEW API)
    # ============================================================================

    def contract_with_compiled_strategy(self, qctn, circuit_states_list, measure_input_list, measure_is_matrix=True):
        """
        Contract using compiled strategy (auto-selected based on mode).
        
        This is the NEW recommended API that automatically selects the best strategy.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states.
            measure_input (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
        
        Returns:
            Backend tensor: Result of the contraction.
        """

        circuit_states = circuit_states_list
        states_shape = tuple([s.shape if s is not None else () for s in circuit_states_list])
    
        measure_shape = tuple([m.shape if m is not None else () for m in measure_input_list])
        measure_input = measure_input_list

        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if not hasattr(qctn, cache_key):
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
        # Pass cores weights directly to support TNTensor
        cores_dict = {name: qctn.cores_weights[name] for name in qctn.cores}
        
        # Execute
        result = compute_fn(cores_dict, circuit_states, measure_input)
        
        return result

    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list, measure_input_list, measure_is_matrix=True) -> Tuple:
        """
        Contract using compiled strategy and compute gradients.
        
        This is the NEW recommended API for gradient computation.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_list (array or list, optional): Circuit input states.
            measure_input_list (array or list, optional): Measurement input.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix.
        
        Returns:
            tuple: (loss, gradients)
        """

        circuit_states = circuit_states_list
        states_shape = tuple([s.shape if s is not None else () for s in circuit_states_list])

        measure_shape = tuple([m.shape if m is not None else () for m in measure_input_list])
        measure_input = measure_input_list
        
        shapes_info = {
            'circuit_states_shapes': states_shape,
            'measure_shapes': measure_shape,
            'measure_is_matrix': measure_is_matrix
        }
        
        # Check cache
        cache_key = f'_compiled_strategy_{self.strategy_mode}_{states_shape}_{measure_shape}_{measure_is_matrix}'
        
        if not hasattr(qctn, cache_key):
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

        # Prepare tensors for gradient calculation
        # We need to separate tensors (which require grad) from scales (constants)
        raw_core_tensors = []
        core_scales = []
        
        for c_name in qctn.cores:
            c = qctn.cores_weights[c_name]
            if isinstance(c, TNTensor):
                raw_core_tensors.append(c.tensor)
                core_scales.append(c.scale)
            else:
                raw_core_tensors.append(c)
                core_scales.append(1.0)

        # Define loss function
        def loss_fn(*core_tensors_args):
            # Reconstruct cores_dict with TNTensors or raw tensors
            reconstructed_cores_dict = {}

            

            for i, name in enumerate(qctn.cores):
                tensor = core_tensors_args[i]
                scale = core_scales[i]
                
                # Check if we should wrap in TNTensor
                # We do this if the original was TNTensor (scale != 1.0 is a heuristic, but better to check original type)
                # But here we simplified lists. Let's assume if we have a scale, we wrap.
                # Actually, compute_fn might EXPECT TNTensor if strategy was compiled/checked against it?
                # GreedyStrategy is dynamic.
                
                if isinstance(qctn.cores_weights[name], TNTensor):
                    reconstructed_cores_dict[name] = TNTensor(tensor, scale)
                else:
                    reconstructed_cores_dict[name] = tensor

            
            # Execute contraction
            # compute_fn will handle TNTensors internally (auto-scaling intermediate results)
            result = compute_fn(reconstructed_cores_dict, circuit_states, measure_input)
            
            # Result might be TNTensor or raw tensor
            if isinstance(result, TNTensor):
                res_tensor = result.tensor
                res_scale = result.scale
            else:
                res_tensor = result
                res_scale = 1.0
            
            # Compute Cross Entropy loss
            # Target is all ones (maximizing probability)
            target = self.backend.ones(res_tensor.shape, dtype=res_tensor.dtype)
            
            # Avoid log(0)
            res_tensor = self.backend.clamp(res_tensor, min=1e-10)
            log_result = self.backend.log(res_tensor)

            # print(f"res_tensor : {res_tensor}, res_scale: {res_scale}")

            # Add log(scale) for correct loss value (log(P*S) = log(P) + log(S))
            # log(S) is constant w.r.t parameters, so gradients are correct
            detached_scale = self.backend.detach(res_scale)
            
            # # Handle float/scalar scale for log
            # import torch
            if isinstance(detached_scale, (int, float)):
                 log_scale = np.log(detached_scale)
            else:
                 # Check if 0-dim tensor
                 if detached_scale.ndim == 0:
                      log_scale = self.backend.log(detached_scale)
                 else:
                      log_scale = self.backend.log(detached_scale)

            # print('log_scale', log_scale)

            log_total = log_result + log_scale
            
            return -self.backend.mean(target * log_total)
        
        # Compute gradients
        # We want gradients with respect to all cores
        argnums = list(range(len(raw_core_tensors)))
        
        # Create value_and_grad function
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        
        # Execute
        loss, grads = value_and_grad_fn(*raw_core_tensors)
        

        # grads = [grads[i] / core_scales[i] for i in range(len(core_scales))]

        # tmp = {i: (grads[i], core_scales[i]) for i in range(len(grads))}
        # print(f"grads : {tmp}")
        # print(f"scale : {{i: core_scales[i] for i in range(len(core_scales))}}")

        return loss, grads
        
        # Create value_and_grad function
        value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
        
        # Execute
        loss, grads = value_and_grad_fn(*core_tensors)
        
        return loss, grads

    # ============================================================================
    # Probability Calculation Methods
    # ============================================================================

    def calculate_full_probability(self, qctn, circuit_states_list, measure_input_list):
        """
        Calculate the full probability of observing a specific bitstring.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (complete).
            
        Returns:
            Backend tensor: The calculated probability.
        """
        return self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=measure_input_list, 
            measure_is_matrix=True
        )

    def calculate_marginal_probability(self, qctn, circuit_states_list, measure_input_list, qubit_indices: List[int]):
        """
        Calculate the marginal probability of a subset of qubits being in a specific state.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (partial).
            qubit_indices (list[int]): Indices of qubits corresponding to measure_input_list.
            
        Returns:
            Backend tensor: The calculated probability (or batch of probabilities).
        """

        if len(qubit_indices) != len(measure_input_list):
            raise ValueError("Length of qubit_indices must match length of measure_input_list")
        
        dim = 1
        for m in measure_input_list:
            if m is not None:
                dim = m.shape[-1]
                break

        full_measure_input_list = []
        
        # Create Identity matrix
        ident = self.backend.eye(dim)
        # If measure_input_list has batch dim, ident should broadcast or match?
        # Usually measure_input_list elements are (B, K, K) or (K, K).
        # We assume (B, K, K) or compatible.
        # If we need batch dim for identity, we can add it later or rely on broadcasting.
        # But contract_with_compiled_strategy expects consistent batch dims if present.
        # Let's check the first element of measure_input_list to see if it has batch dim.
        has_batch = False
        batch_size = 1
        if len(measure_input_list) > 0:
            if measure_input_list[0].ndim == 3:
                has_batch = True
                batch_size = measure_input_list[0].shape[0]
                ident = self.backend.unsqueeze(ident, 0)
                ident = self.backend.expand(ident, batch_size, -1, -1)

        for i in range(qctn.nqubits):
            if i in qubit_indices:
                idx = qubit_indices.index(i)
                full_measure_input_list.append(measure_input_list[idx])
            else:
                full_measure_input_list.append(ident)
        
        return self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=full_measure_input_list, 
            measure_is_matrix=True
        )

    def calculate_conditional_probability(self, qctn, circuit_states_list, measure_input_list, 
                                          qubit_indices: List[int], target_indices: List[int]):
        """
        Calculate the conditional probability P(target | condition).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network.
            circuit_states_list (list): List of circuit input states.
            measure_input_list (list): List of measurement input matrices (covering target + condition).
            qubit_indices (list[int]): Indices of qubits corresponding to measure_input_list.
            target_indices (list[int]): Indices of target qubits (subset of qubit_indices).
            
        Returns:
            Backend tensor: The calculated conditional probability.
        """
        # Check inputs
        if len(qubit_indices) != len(measure_input_list):
            raise ValueError("Length of qubit_indices must match length of measure_input_list")
        
        dim = 1
        for m in measure_input_list:
            if m is not None:
                dim = m.shape[-1]
                break
        # Create Identity matrix (B, K, K)
        ident = self.backend.eye(dim)

        has_batch = False
        batch_size = 1
        if len(measure_input_list) > 0:
            if measure_input_list[0].ndim == 3:
                has_batch = True
                batch_size = measure_input_list[0].shape[0]
                ident = self.backend.unsqueeze(ident, 0)
                ident = self.backend.expand(ident, batch_size, -1, -1)

        # Prepare stacked measurements
        # We want output shape (B, 2) -> effectively batch size 2*B? Or B*2?
        # The user requested: "change shape to B*2*K*K".
        # Index 0: Original (Joint P(A,B))
        # Index 1: Identity on Target (Marginal P(B))
        
        full_measure_input_list = []
        
        for i in range(qctn.nqubits):
            # Prepare tensor of shape (B, 2, K, K)
            
            if i in qubit_indices:
                idx = qubit_indices.index(i)
                measure_tensor = measure_input_list[idx] # (B, K, K)
                
                if i in target_indices:
                    # Target qubit: [Measure, Identity]
                    # Stack along dim 1
                    stacked = self.backend.stack([measure_tensor, ident], dim=1) # (B, 2, K, K)
                else:
                    # Condition qubit: [Measure, Measure]
                    stacked = self.backend.stack([measure_tensor, measure_tensor], dim=1) # (B, 2, K, K)
            else:
                # Unused qubit: [Identity, Identity]
                stacked = self.backend.stack([ident, ident], dim=1) # (B, 2, K, K)
            
            full_measure_input_list.append(stacked)
        
        # Contract
        # The engine's einsum strategy should handle the extra dimension '2' via broadcasting '...'
        # Result shape should be (B, 2)
        result = self.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list, 
            measure_input_list=full_measure_input_list, 
            measure_is_matrix=True
        )
        
        # Calculate conditional probability
        # result[:, 0] is Joint P(A, B)
        # result[:, 1] is Marginal P(B)
        # P(A|B) = P(A, B) / P(B)
        
        prob_joint = result[:, 0]
        prob_condition = result[:, 1]
        
        epsilon = 1e-10
        return prob_joint / (prob_condition + epsilon)

    # ============================================================================
    # Sampling Methods
    # ============================================================================

    def sample(self, qctn, circuit_states_list, num_samples, dim):
        """
        Sample bitstrings from the quantum circuit.
        
        Args:
            qctn: QCTN object
            circuit_states_list: List of input states
            num_samples: Number of samples (batch size)
            dim: Dimension of each qubit (e.g. 2)
            
        Returns:
            res: Tensor of shape (num_samples, nqubits) containing sampled indices.
        """
        
        # Initialize measure_input_list with Identities
        # Shape: (B, dim, dim)
        ident = self.backend.eye(dim)
        ident_batch = self.backend.unsqueeze(ident, 0)
        ident_batch = self.backend.expand(ident_batch, num_samples, -1, -1)
        
        measure_input_list = [self.backend.clone(ident_batch) for _ in range(qctn.nqubits)]
        
        # Create result tensor for storing sampled indices (integer type)
        # For PyTorch: torch.long, For JAX: jnp.int32
        res_list = []
        
        for i in range(qctn.nqubits):
            measure_input_list[i] = None

            rho_i = self.contract_with_compiled_strategy(
                qctn, 
                circuit_states_list=circuit_states_list, 
                measure_input_list=measure_input_list, 
                measure_is_matrix=True
            )

            print(f"[EngineSiamese] Sampling qubit {i}, rho_i shape: {rho_i.shape}")

            # Extract diagonal and normalize
            probs_unnorm = self.backend.diagonal(rho_i, dim1=-2, dim2=-1) # (B, dim)
            
            # Handle negative values (numerical noise) and normalize
            probs_unnorm = self.backend.clamp(probs_unnorm, min=0.0)
            norms = self.backend.sum(probs_unnorm, dim=-1, keepdim=True)
            probs = probs_unnorm / (norms + 1e-10)

            print(f"[EngineSiamese] Qubit {i}, probs sum (should be 1): {self.backend.sum(probs, dim=-1)}, probs: {probs}")
            
            # Sample
            sampled_indices = self.backend.multinomial(probs, num_samples=1) # (B, 1)
            sampled_indices_1d = self.backend.squeeze(sampled_indices, dim=-1)
            res_list.append(sampled_indices_1d)
            
            # Update measure_input_list[i] with projector |k><k|
            new_measure = self.backend.zeros((num_samples, dim, dim), dtype=probs.dtype)
            batch_indices = self.backend.arange(num_samples)
            new_measure[batch_indices, sampled_indices_1d, sampled_indices_1d] = 1.0
            
            measure_input_list[i] = new_measure
        
        # Stack all sampled indices into a single tensor of shape (num_samples, nqubits)
        res = self.backend.stack(res_list, dim=1)
        
        return res
