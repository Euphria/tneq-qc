from __future__ import annotations
from config import Configuration
import itertools
import opt_einsum
import jax
import jax.numpy as jnp

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
        cores_weights = qctn.cores_weights

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

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)
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
        cores_weights = qctn.cores_weights

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

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        inputs_cores = [inputs] + [cores_weights[core_name] for core_name in cores_name]
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_vector_inputs(qctn, inputs):
        """
        Contract the given QCTN with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (list[jnp.ndarray]): The vector inputs for the contraction operation.

        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

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

        tensor_shapes = [v.shape for v in inputs] + [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        print(f'QCTN: {qctn.circuit}')
        print(f'Einsum Equation: {einsum_equation}')
        print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        inputs_cores = inputs + [cores_weights[core_name] for core_name in cores_name]
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_QCTN(qctn, target_qctn, initialization_mode=False):
        """
        Contract the given QCTN with a target QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network for contraction.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        target_input_ranks, target_adjacency_matrix, target_output_ranks = target_qctn.circuit
        target_cores_name = target_qctn.cores
        target_cores_weights = target_qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''

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

        target_adjacency_matrix_for_interaction = target_adjacency_matrix.copy()
        for element in QCTNHelper.jax_triu_ndindex(len(target_cores_name)):
            i, j = element
            if target_adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(target_adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                target_adjacency_matrix[i, j] = connection_symbols
                target_adjacency_matrix[j, i] = connection_symbols

        input_symbols_stack = []
        output_symbols_stack = []

        # self.einsum_equation_lefthand
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
        
        # target.einsum_equation_lefthand
        for idx, _ in enumerate(target_cores_name):
            for _ in target_input_ranks[idx]:
                target_einsum_equation_lefthand += input_symbols_stack.pop(0)

            target_einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(target_adjacency_matrix[idx])))

            for _ in target_output_ranks[idx]:
                target_einsum_equation_lefthand += output_symbols_stack.pop(0)
            
            target_einsum_equation_lefthand += ','

        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name] + \
                        [target_cores_weights[core_name].shape for core_name in target_cores_name]


        for core_name in cores_name:
            print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')

        for core_name in target_cores_name:
            print(f'Target Core: {core_name}, Shape: {target_cores_weights[core_name].shape}')
        
        print(f'QCTN: {qctn.circuit}')
        print(f'Target QCTN: {target_qctn.circuit}')
        print(f'Einsum Equation: {einsum_equation}')
        print(f'Tensor Shapes: {tensor_shapes}')

        qctn.einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(qctn.einsum_expr)

        if initialization_mode:
            # In initialization mode, we do not actually contract the target QCTN
            return qctn.einsum_expr

        inputs_cores = [cores_weights[core_name] for core_name in cores_name] + \
                       [target_cores_weights[core_name] for core_name in target_cores_name]
        
        retracted_QCTN = jit_retraction(*inputs_cores)

        return retracted_QCTN

    @staticmethod
    def contract_with_QCTN_for_gradient(qctn, target_qctn):
        """
        We use JAX's autograd to compute the core gradient.
        """

        cores_name = qctn.cores
        cores_weights = qctn.cores_weights
        target_cores_name = target_qctn.cores
        target_cores_weights = target_qctn.cores_weights
        
        inputs_cores = [cores_weights[core_name] for core_name in cores_name] + \
            [target_cores_weights[core_name] for core_name in target_cores_name]

        def mse_loss_fn(*inputs_cores):
            retracted_QCTN = qctn.einsum_expr(*inputs_cores)
            return jnp.mean((retracted_QCTN - 1.0) ** 2) 

        if qctn.einsum_expr is None:
            ContractorOptEinsum.contract_with_QCTN(qctn, target_qctn, initialization_mode=True)
            argnums = list(range(len(cores_name)))
            qctn.jit_retraction_with_QCTN_value_gradient = jax.jit(jax.value_and_grad(mse_loss_fn, 
                                                                                      argnums=argnums))

        loss, grad_cores = qctn.jit_retraction_with_QCTN_value_gradient(*inputs_cores)
        print(f'Loss: {loss}')

        return loss, grad_cores