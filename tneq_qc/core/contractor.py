"""
Contractor module for generating tensor contraction expressions and managing strategies.

This module provides:
- ContractionStrategy: Abstract base class for contraction strategies
- Concrete strategies: EinsumStrategy, MPSChainStrategy
- StrategyCompiler: Compiles and selects optimal strategy based on mode
"""

from __future__ import annotations
import itertools
import opt_einsum
from typing import Tuple, List, Optional, Union, Callable, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# Abstract Strategy Interface
# ============================================================================

class ContractionStrategy(ABC):
    """Abstract base class for contraction strategies"""
    
    @abstractmethod
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        Check if the network structure is compatible with this strategy.
        
        Args:
            qctn: QCTN object
            shapes_info: dict, containing circuit_states_shapes, measure_shapes etc.
        
        Returns:
            bool: Whether it is compatible
        """
        pass
    
    @abstractmethod
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        Generate computation function.
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information
            backend: Backend
        
        Returns:
            Callable: compute_fn(cores_dict, circuit_states, measure_matrices)
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        Estimate computation cost (FLOPs).
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information
        
        Returns:
            float: Estimated FLOPs
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================

class EinsumStrategy(ContractionStrategy):
    """Strategy using opt_einsum directly (Fast mode)"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """einsum can handle any structure"""
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        Return computation function using opt_einsum.
        
        Use existing build_with_self_expression to generate einsum expression.
        """
        # Get shape info
        circuit_states_shapes = shapes_info.get('circuit_states_shapes')
        measure_shapes = shapes_info.get('measure_shapes')
        measure_is_matrix = shapes_info.get('measure_is_matrix', True)
        
        # Generate einsum expression
        einsum_eq, tensor_shapes = self.build_with_self_expression(
            qctn, circuit_states_shapes, measure_shapes, measure_is_matrix
        )
        
        # Create optimized expression
        expr = self.create_contract_expression(einsum_eq, tensor_shapes, optimize='auto')
        
        def compute_fn(cores_dict, circuit_states, measure_matrices):
            """
            Compute using einsum expression.
            
            Args:
                cores_dict: {core_name: core_tensor} dictionary
                circuit_states: List of circuit input states
                measure_matrices: List of measurement matrices
            
            Returns:
                Contraction result
            """
            # Prepare tensors in order
            tensors = []
            
            # Add circuit_states
            if circuit_states is not None:
                if isinstance(circuit_states, list):
                    tensors.extend(circuit_states)
                else:
                    tensors.append(circuit_states)
            
            # Add cores
            for core_name in qctn.cores:
                tensors.append(cores_dict[core_name])
            
            # Add measure_matrices
            if measure_matrices is not None:
                if isinstance(measure_matrices, list):
                    tensors.extend(measure_matrices)
                else:
                    tensors.append(measure_matrices)
            
            # Add inverse cores
            for core_name in reversed(qctn.cores):
                tensors.append(cores_dict[core_name])
            
            # Add circuit_states again (conjugate side)
            if circuit_states is not None:
                if isinstance(circuit_states, list):
                    tensors.extend(circuit_states)
                else:
                    tensors.append(circuit_states)
            
            # Execute expression
            jit_fn = backend.jit_compile(expr)
            return backend.execute_expression(jit_fn, *tensors)
        
        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """Estimate cost using opt_einsum"""
        circuit_states_shapes = shapes_info.get('circuit_states_shapes')
        measure_shapes = shapes_info.get('measure_shapes')
        measure_is_matrix = shapes_info.get('measure_is_matrix', True)
        
        einsum_eq, tensor_shapes = self.build_with_self_expression(
            qctn, circuit_states_shapes, measure_shapes, measure_is_matrix
        )
        
        # try:
        #     # Estimate cost using opt_einsum
        #     path_info = opt_einsum.contract_path(einsum_eq, *tensor_shapes, optimize='auto')
        #     return float(path_info[1].opt_cost)
        # except:
        #     # Return a large value if estimation fails
        #     return float('inf')

        # Estimate cost using opt_einsum
        path_info = opt_einsum.contract_path(einsum_eq, *tensor_shapes, optimize='auto')
        return float(path_info[1].opt_cost)
    
    @property
    def name(self) -> str:
        return "einsum_default"

    @staticmethod
    def build_core_only_expression(qctn) -> Tuple[str, List]:
        """
        Build einsum expression for contracting cores only (no inputs).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .qctn import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_inputs_expression(qctn, inputs_shape) -> Tuple[str, List]:
        """
        Build einsum expression for contracting with single input tensor.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shape (tuple): Shape of the input tensor.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .qctn import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand},{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [inputs_shape] + [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_vector_inputs_expression(qctn, inputs_shapes: List) -> Tuple[str, List]:
        """
        Build einsum expression for contracting with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shapes (list): List of input tensor shapes.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .qctn import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += ','
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand}{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = inputs_shapes + [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_qctn_expression(qctn, target_qctn) -> Tuple[str, List]:
        """
        Build einsum expression for contracting two QCTNs together.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        target_input_ranks, target_adjacency_matrix, target_output_ranks = target_qctn.circuit
        target_cores_name = target_qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .qctn import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        target_adjacency_matrix_for_interaction = target_adjacency_matrix.copy()
        for element in QCTNHelper.jax_triu_ndindex(len(target_cores_name)):
            i, j = element
            if target_adjacency_matrix_for_interaction[i, j]:
                connection_num = len(target_adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                target_adjacency_matrix[i, j] = connection_symbols
                target_adjacency_matrix[j, i] = connection_symbols

        input_symbols_stack = []
        output_symbols_stack = []

        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += ','

        for idx, _ in enumerate(target_cores_name):
            for _ in target_input_ranks[idx]:
                target_einsum_equation_lefthand += input_symbols_stack.pop(0)

            target_einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(target_adjacency_matrix[idx])))

            for _ in target_output_ranks[idx]:
                target_einsum_equation_lefthand += output_symbols_stack.pop(0)

            target_einsum_equation_lefthand += ','

        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name] + \
                        [target_qctn.cores_weights[core_name].shape for core_name in target_cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_self_expression(qctn, circuit_states_shape=None, measure_shape=None, measure_is_matrix=False) -> Tuple[str, List]:
        """
        Build einsum expression for contracting QCTN with itself (hermitian conjugate).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states_shape (tuple or tuple of tuples, optional): Shape(s) of circuit states input.
                Can be a single shape tuple or tuple of shape tuples for list inputs.
            measure_shape (tuple or tuple of tuples, optional): Shape(s) of measurement input.
                Can be a single shape tuple or tuple of shape tuples for list inputs.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """

        # Determine if we have list inputs
        is_states_list = isinstance(circuit_states_shape, tuple) and circuit_states_shape and isinstance(circuit_states_shape[0], tuple)
        is_measure_list = isinstance(measure_shape, tuple) and measure_shape and isinstance(measure_shape[0], tuple)
        
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .qctn import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        input_symbols_stack = []
        output_symbols_stack = []

        equation_list = []
        new_symbol_mapping = {}

        for idx, _ in enumerate(cores_name):
            core_equation = ""

            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            core_equation += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1

            # TODO: use better strategy
            ll = core_equation[:2]
            rr = core_equation[2:]
            # sort string characters
            ll = list(ll)
            ll.sort()
            rr = list(rr)
            rr.sort()
            core_equation = "".join(ll + rr[::-1])

            einsum_equation_lefthand += core_equation
            equation_list.append(core_equation)

        middle_block_list = []
        middle_symbols_mapping = {
            char: char for char in output_symbols_stack
        }
        batch_symbol = ''
        if measure_shape is not None:
            # Add batch size dimension
            batch_symbol = opt_einsum.get_symbol(symbol_id)
            symbol_id += 1
            
            middle_block_list = []
            for char in output_symbols_stack:
                symbol = opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

                middle_symbols_mapping[char] = symbol

                middle_block_list += [batch_symbol + char + symbol]
            # swap last two 
            if len(middle_block_list) >=2:
                middle_block_list = middle_block_list[:-2] + middle_block_list[-2:][::-1]

        # print('output_symbols_stack', output_symbols_stack)
        # print('middle_block_list', middle_block_list)

        real_output_symbols_stack = []
        inv_equation_list = []
        for core_equation in equation_list[::-1]:
            new_equation = ""
            for char in core_equation:
                if char in output_symbols_stack:
                    new_equation += middle_symbols_mapping[char]
                else:
                    if char in new_symbol_mapping:
                        symbol = new_symbol_mapping[char]
                    else:
                        symbol = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                        new_symbol_mapping[char] = symbol
                        # print(f"mapping {char} to {symbol}")

                        if char in input_symbols_stack:
                            real_output_symbols_stack.append(symbol)

                    new_equation += symbol

            inv_equation_list.append(new_equation)

        equation_list = equation_list + middle_block_list + inv_equation_list

        einsum_equation_lefthand = ",".join(equation_list)
        
        # Handle circuit_states and measure_input
        if is_states_list:
            circuit_states_symbols = ','.join(input_symbols_stack)
            output_states_symbols = ''
            for char in circuit_states_symbols[::-1]:
                output_states_symbols += char if char==',' else new_symbol_mapping[char]
            # output_states_symbols = ','.join(real_output_symbols_stack)
        else:
            circuit_states_symbols = ''.join(input_symbols_stack)
            output_states_symbols = ''
            for char in circuit_states_symbols[::-1]:
                output_states_symbols += new_symbol_mapping[char]
            # output_states_symbols = ''.join(real_output_symbols_stack)

        # Build equation parts
        left_parts = []
        
        # Add circuit_states
        if circuit_states_shape is not None:
            left_parts.append(circuit_states_symbols)
        
        # Add cores equations
        left_parts.append(einsum_equation_lefthand)
        
        # Add conjugate side inputs
        if circuit_states_shape is not None:
            left_parts.append(output_states_symbols)
        
        einsum_equation_lefthand = ",".join(left_parts)

        einsum_equation = f'{einsum_equation_lefthand}->{batch_symbol}'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name]
        inv_tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name[::-1]]

        # Prepare tensor_shapes list
        shapes_list = []
        
        # Add circuit_states shapes
        if circuit_states_shape is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shape))
            else:
                shapes_list.append(circuit_states_shape)
        
        # Add core shapes
        shapes_list.extend(tensor_shapes)

        # Add measure_input shapes
        if measure_shape is not None:
            if is_measure_list:
                shapes_list.extend(list(measure_shape))
            else:
                shapes_list.append(measure_shape)
        
        # Add inverse core shapes
        shapes_list.extend(inv_tensor_shapes)
        
        # Add conjugate side shapes
        if circuit_states_shape is not None:
            if is_states_list:
                shapes_list.extend(list(circuit_states_shape))
            else:
                shapes_list.append(circuit_states_shape)
        
        tensor_shapes = shapes_list

        return einsum_equation, tensor_shapes

    @staticmethod
    def create_contract_expression(einsum_equation: str, tensor_shapes: List, optimize='auto'):
        """
        Create optimized contraction expression using opt_einsum.
        
        Args:
            einsum_equation (str): The einsum equation string.
            tensor_shapes (list): List of tensor shapes.
            optimize (str or bool): Optimization strategy for opt_einsum.
        
        Returns:
            opt_einsum.ContractExpression: Optimized contraction expression.
        """
        from ..config import Configuration
        print('einsum_equation', einsum_equation)
        print('tensor_shapes', tensor_shapes)

        return opt_einsum.contract_expression(
            einsum_equation, 
            *tensor_shapes, 
            optimize=optimize if optimize != 'auto' else Configuration.opt_einsum_optimize
        )


class MPSChainStrategy(ContractionStrategy):
    """Optimization strategy for MPS chain structure (Balanced/Full mode)"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        Check if it is a chain structure.
        
        Currently simplified implementation: returns True directly.
        More strict topology checks can be added in the future.
        """
        # TODO: Implement stricter chain structure check
        # 1. Check if topology is a chain
        # 2. Check connection method of each core
        # 3. Check if input/output dimensions meet expectations
        
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        Return computation function for MPS chain contraction.
        
        This is similar to contract_with_std_graph implementation.
        """
        def compute_fn(cores_dict, circuit_states, measure_matrices):
            """
            MPS chain contraction.
            
            Args:
                cores_dict: {core_name: core_tensor} dictionary
                circuit_states: List of circuit input states
                measure_matrices: List of measurement matrices
            
            Returns:
                Contraction result
            """
            import torch
            
            new_core_dict = {}
            
            # Step 1: Contract cores with circuit_states
            for idx, core_name in enumerate(qctn.cores):
                core_tensor = cores_dict[core_name]
                
                if idx == 0:
                    # First core contracts two states
                    state1 = circuit_states[0]
                    state2 = circuit_states[1]
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)
                else:
                    # Other cores contract one state
                    state = circuit_states[idx + 1]
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                
                new_core_dict[core_name] = contracted
            
            print('new_core_dict', [(core_name, new_core_dict[core_name].shape) for core_name in new_core_dict])

            # Step 2: Chain contraction with measurements
            n = len(new_core_dict)
            
            if n == 1:
                core_tensor = new_core_dict[qctn.cores[0]]
                measure_matrix_1 = measure_matrices[0]
                measure_matrix_2 = measure_matrices[1]
                contracted = torch.einsum('ka,zkl,zab,lb->z', 
                                         core_tensor, measure_matrix_1, 
                                         measure_matrix_2, core_tensor)
                return contracted

            for idx in range(n):
                
                if idx == 0:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix = measure_matrices[idx]
                    contracted = torch.einsum('ka,zkl,lb->zab', 
                                             core_tensor, measure_matrix, core_tensor)
                    
                elif idx < n - 1:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix = measure_matrices[idx]
                    contracted = torch.einsum('zab,akc,zkl,bld->zcd', 
                                             contracted, core_tensor, 
                                             measure_matrix, core_tensor)
                else:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix_1 = measure_matrices[idx]
                    measure_matrix_2 = measure_matrices[idx + 1]
                    contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', 
                                             contracted, core_tensor, 
                                             measure_matrix_1, measure_matrix_2, core_tensor)
                
                print('contract step', idx, contracted.shape)

            return contracted
        
        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        Estimate cost of MPS chain contraction.
        
        Currently simplified implementation: returns a small fixed value.
        Future implementation can estimate precisely based on dimensions of each einsum step.
        """
        # TODO: Implement precise FLOPs estimation
        # Calculate based on dimensions of each einsum step
        
        circuit_states_shapes = shapes_info.get('circuit_states_shapes', [])
        measure_shapes = shapes_info.get('measure_shapes', [])
        
        # Simplified estimation: assume MPS strategy is usually better than einsum
        total_flops = 1e6  # Return a small fixed value
        
        return total_flops
    
    @property
    def name(self) -> str:
        return "mps_chain"


# ============================================================================
# Strategy Compiler
# ============================================================================

class StrategyCompiler:
    """Strategy compiler, responsible for selecting and compiling the optimal strategy"""
    
    # Strategy list for three modes
    MODES = {
        'fast': ['einsum_default'],
        'balanced': ['mps_chain'],
        'full': ['mps_chain']
    }
    
    def __init__(self, mode: str = 'fast'):
        """
        Initialize compiler
        
        Args:
            mode: 'fast', 'balanced', or 'full'
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(self.MODES.keys())}")
        
        self.mode = mode
        self.strategies: Dict[str, ContractionStrategy] = {}
        self._register_strategies()
    
    def _register_strategies(self):
        """Register all available strategies"""
        # Fast mode
        self.strategies['einsum_default'] = EinsumStrategy()
        
        # Balanced/Full mode
        self.strategies['mps_chain'] = MPSChainStrategy()
        
        # TODO: Add more strategies in the future
        # self.strategies['tree_contraction'] = TreeContractionStrategy()
        # self.strategies['greedy_path'] = GreedyPathStrategy()
    
    def compile(self, qctn, shapes_info: Dict[str, Any], backend) -> Tuple[Callable, str, float]:
        """
        Compile: Select optimal strategy and return computation function
        
        Compilation process:
        1. Check structure compatibility
        2. Estimate cost
        3. Generate computation function
        4. Select strategy with lowest cost
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information dict
            backend: Computation backend
        
        Returns:
            tuple: (compute_fn, strategy_name, estimated_cost)
        """
        # Get strategy list for current mode
        strategy_names = self.MODES[self.mode]
        
        candidates = []
        
        print(f"[Compiler] Mode: {self.mode}, Testing {len(strategy_names)} strategies...")
        
        # Iterate over all candidate strategies
        for name in strategy_names:
            strategy = self.strategies[name]
            
            # 1. Check compatibility
            # try:
            #     is_compatible = strategy.check_compatibility(qctn, shapes_info)
            #     print(f"  [{name}] Compatibility: {is_compatible}")
            #     
            #     if not is_compatible:
            #         continue
            # except Exception as e:
            #     print(f"  [{name}] Compatibility check failed: {e}")
            #     continue

            is_compatible = strategy.check_compatibility(qctn, shapes_info)
            print(f"  [{name}] Compatibility: {is_compatible}")
            
            if not is_compatible:
                continue
            
            # 2. Estimate cost
            # try:
            #     cost = strategy.estimate_cost(qctn, shapes_info)
            #     print(f"  [{name}] Estimated cost: {cost:.2e} FLOPs")
            # except Exception as e:
            #     print(f"  [{name}] Cost estimation failed: {e}")
            #     cost = float('inf')

            cost = strategy.estimate_cost(qctn, shapes_info)
            print(f"  [{name}] Estimated cost: {cost:.2e} FLOPs")
            
            # 3. Generate computation function
            # try:
            #     compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
            # except Exception as e:
            #     print(f"  [{name}] Function generation failed: {e}")
            #     continue

            compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
            
            candidates.append({
                'name': name,
                'strategy': strategy,
                'compute_fn': compute_fn,
                'cost': cost
            })
        
        # Select strategy with lowest cost
        if not candidates:
            raise RuntimeError("No compatible strategy found!")
        
        best = min(candidates, key=lambda x: x['cost'])
        print(f"[Compiler] Selected strategy: {best['name']} (cost: {best['cost']:.2e})")
        
        return best['compute_fn'], best['name'], best['cost']
    
    def register_custom_strategy(self, strategy: ContractionStrategy, modes: List[str]):
        """
        Register custom strategy
        
        Args:
            strategy: Strategy instance
            modes: Which modes to register to, e.g. ['balanced', 'full']
        """
        self.strategies[strategy.name] = strategy
        
        for mode in modes:
            if mode in self.MODES:
                if strategy.name not in self.MODES[mode]:
                    self.MODES[mode].append(strategy.name)



