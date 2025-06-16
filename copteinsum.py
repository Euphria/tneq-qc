from __future__ import annotations
from config import Configuration
import itertools
import opt_einsum
import jax

class ContractorOptEinsum:
    """
    ContractorOptEinsum class for optimized tensor contraction using opt_einsum.
    
    This class provides methods to contract tensors using the opt_einsum library,
    which is optimized for performance and memory efficiency.
    """

    # tenmul_qc.py
    @staticmethod
    def contract_core_only(qctn):
        """
        Contract the given QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weigts

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
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

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]  # Remove the last comma
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        print(f'QCTN: {qctn.circuit}')
        print(f'Einsum Equation: {einsum_equation}')
        print(f'Tensor Shapes: {tensor_shapes}')

        einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(einsum_expr)
        retracted_QCTN = jit_retraction(*[cores_weights[core_name] for core_name in cores_name])

        return retracted_QCTN
    
    @staticmethod
    def contract_with_inputs(qctn, inputs):
        """
        Contract the given QCTN with specified inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (jnp.ndarray): The inputs for the contraction operation.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weigts

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
        
        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
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

        tensor_shapes = [inputs.shape] + [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        print(f'QCTN: {qctn.circuit}')
        print(f'Einsum Equation: {einsum_equation}')
        print(f'Tensor Shapes: {tensor_shapes}')

        einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(einsum_expr)

        inputs_cores = [inputs] + [cores_weights[core_name] for core_name in cores_name]
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN