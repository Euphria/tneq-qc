"""
Contractor module for generating tensor contraction expressions using opt_einsum.

This module separates the contraction logic from backend execution:
- Contractor: Generates einsum expressions and manages contraction strategies
- Backend: Executes the expressions using JAX, PyTorch, or other frameworks
"""

from __future__ import annotations
import itertools
import opt_einsum
from typing import Tuple, List, Optional, Union
import numpy as np


class TensorContractor:
    """
    TensorContractor class for generating optimized tensor contraction expressions.
    
    This class uses opt_einsum to generate contraction expressions but does not
    execute them. The execution is delegated to backend implementations.
    """

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

        from .tenmul_qc import QCTNHelper
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

        from .tenmul_qc import QCTNHelper
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

        from .tenmul_qc import QCTNHelper
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

        from .tenmul_qc import QCTNHelper
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

        from .tenmul_qc import QCTNHelper
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
