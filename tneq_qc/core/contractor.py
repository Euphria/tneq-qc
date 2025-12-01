"""
Contractor module for generating tensor contraction expressions and managing strategies.

This module provides:
- TensorContractor: Original einsum expression generation (kept for compatibility)
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
    """收缩策略抽象基类"""
    
    @abstractmethod
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        检查网络结构是否符合该策略要求
        
        Args:
            qctn: QCTN对象
            shapes_info: dict，包含 circuit_states_shapes, measure_shapes 等
        
        Returns:
            bool: 是否兼容
        """
        pass
    
    @abstractmethod
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        生成计算函数
        
        Args:
            qctn: QCTN对象
            shapes_info: 形状信息
            backend: 后端
        
        Returns:
            Callable: 计算函数 compute_fn(cores_dict, circuit_states, measure_matrices)
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        估算计算代价（FLOPs）
        
        Args:
            qctn: QCTN对象
            shapes_info: 形状信息
        
        Returns:
            float: 估算的 FLOPs
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================

class EinsumStrategy(ContractionStrategy):
    """直接使用 opt_einsum 的策略（Fast mode）"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """einsum 可以处理任何结构"""
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        返回使用 opt_einsum 的计算函数
        
        使用现有的 build_with_self_expression 生成 einsum 表达式
        """
        # 获取形状信息
        circuit_states_shapes = shapes_info.get('circuit_states_shapes')
        measure_shapes = shapes_info.get('measure_shapes')
        measure_is_matrix = shapes_info.get('measure_is_matrix', True)
        
        # 生成 einsum 表达式
        contractor = TensorContractor()
        einsum_eq, tensor_shapes = contractor.build_with_self_expression(
            qctn, circuit_states_shapes, measure_shapes, measure_is_matrix
        )
        
        # 创建优化的表达式
        expr = contractor.create_contract_expression(einsum_eq, tensor_shapes, optimize='auto')
        
        def compute_fn(cores_dict, circuit_states, measure_matrices):
            """
            使用 einsum 表达式计算
            
            Args:
                cores_dict: {core_name: core_tensor} 字典
                circuit_states: 电路输入态列表
                measure_matrices: 测量矩阵列表
            
            Returns:
                收缩结果
            """
            # 按顺序准备张量
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
        """使用 opt_einsum 估算代价"""
        circuit_states_shapes = shapes_info.get('circuit_states_shapes')
        measure_shapes = shapes_info.get('measure_shapes')
        measure_is_matrix = shapes_info.get('measure_is_matrix', True)
        
        contractor = TensorContractor()
        einsum_eq, tensor_shapes = contractor.build_with_self_expression(
            qctn, circuit_states_shapes, measure_shapes, measure_is_matrix
        )
        
        try:
            # 使用 opt_einsum 估算代价
            path_info = opt_einsum.contract_path(einsum_eq, *tensor_shapes, optimize='auto')
            return float(path_info[1].opt_cost)
        except:
            # 如果估算失败，返回一个较大的值
            return float('inf')
    
    @property
    def name(self) -> str:
        return "einsum_default"


class MPSChainStrategy(ContractionStrategy):
    """针对 MPS 链式结构的优化策略（Balanced/Full mode）"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        检查是否为链式结构
        
        目前简化实现：直接返回 True
        未来可以添加更严格的拓扑检查
        """
        # TODO: 实现更严格的链式结构检查
        # 1. 检查拓扑是否为链
        # 2. 检查每个 core 的连接方式
        # 3. 检查输入输出维度是否符合预期
        
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        返回 MPS 链式收缩的计算函数
        
        这是类似 contract_with_std_graph 的实现
        """
        def compute_fn(cores_dict, circuit_states, measure_matrices):
            """
            MPS 链式收缩
            
            Args:
                cores_dict: {core_name: core_tensor} 字典
                circuit_states: 电路输入态列表
                measure_matrices: 测量矩阵列表
            
            Returns:
                收缩结果
            """
            import torch
            
            new_core_dict = {}
            
            # Step 1: Contract cores with circuit_states
            for idx, core_name in enumerate(qctn.cores):
                core_tensor = cores_dict[core_name]
                
                if idx == 0:
                    # 第一个 core 收缩两个 state
                    state1 = circuit_states[0]
                    state2 = circuit_states[1]
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)
                else:
                    # 其他 core 收缩一个 state
                    state = circuit_states[idx + 1]
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                
                new_core_dict[core_name] = contracted
            
            # Step 2: Chain contraction with measurements
            n = len(new_core_dict)
            
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
            
            return contracted
        
        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        估算 MPS 链式收缩的代价
        
        目前简化实现：返回一个较小的固定值
        未来可以基于每一步 einsum 的维度精确估算
        """
        # TODO: 实现精确的 FLOPs 估算
        # 基于每一步 einsum 的维度计算
        
        circuit_states_shapes = shapes_info.get('circuit_states_shapes', [])
        measure_shapes = shapes_info.get('measure_shapes', [])
        
        # 简化估算：假设 MPS 策略通常比 einsum 更优
        total_flops = 1e6  # 返回一个较小的固定值
        
        return total_flops
    
    @property
    def name(self) -> str:
        return "mps_chain"


# ============================================================================
# Strategy Compiler
# ============================================================================

class StrategyCompiler:
    """策略编译器，负责选择和编译最优策略"""
    
    # 三种模式对应的策略列表
    MODES = {
        'fast': ['einsum_default'],
        'balanced': ['einsum_default', 'mps_chain'],
        'full': ['einsum_default', 'mps_chain']
    }
    
    def __init__(self, mode: str = 'fast'):
        """
        初始化编译器
        
        Args:
            mode: 'fast', 'balanced', 或 'full'
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(self.MODES.keys())}")
        
        self.mode = mode
        self.strategies: Dict[str, ContractionStrategy] = {}
        self._register_strategies()
    
    def _register_strategies(self):
        """注册所有可用策略"""
        # Fast mode
        self.strategies['einsum_default'] = EinsumStrategy()
        
        # Balanced/Full mode
        self.strategies['mps_chain'] = MPSChainStrategy()
        
        # TODO: 未来可以添加更多策略
        # self.strategies['tree_contraction'] = TreeContractionStrategy()
        # self.strategies['greedy_path'] = GreedyPathStrategy()
    
    def compile(self, qctn, shapes_info: Dict[str, Any], backend) -> Tuple[Callable, str, float]:
        """
        编译：选择最优策略并返回计算函数
        
        编译过程：
        1. 检查结构兼容性
        2. 估算代价
        3. 生成计算函数
        4. 选择代价最低的策略
        
        Args:
            qctn: QCTN对象
            shapes_info: 形状信息 dict
            backend: 计算后端
        
        Returns:
            tuple: (compute_fn, strategy_name, estimated_cost)
        """
        # 获取当前模式下的策略列表
        strategy_names = self.MODES[self.mode]
        
        candidates = []
        
        print(f"[Compiler] Mode: {self.mode}, Testing {len(strategy_names)} strategies...")
        
        # 遍历所有候选策略
        for name in strategy_names:
            strategy = self.strategies[name]
            
            # 1. 检查兼容性
            try:
                is_compatible = strategy.check_compatibility(qctn, shapes_info)
                print(f"  [{name}] Compatibility: {is_compatible}")
                
                if not is_compatible:
                    continue
            except Exception as e:
                print(f"  [{name}] Compatibility check failed: {e}")
                continue
            
            # 2. 估算代价
            try:
                cost = strategy.estimate_cost(qctn, shapes_info)
                print(f"  [{name}] Estimated cost: {cost:.2e} FLOPs")
            except Exception as e:
                print(f"  [{name}] Cost estimation failed: {e}")
                cost = float('inf')
            
            # 3. 生成计算函数
            try:
                compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
            except Exception as e:
                print(f"  [{name}] Function generation failed: {e}")
                continue
            
            candidates.append({
                'name': name,
                'strategy': strategy,
                'compute_fn': compute_fn,
                'cost': cost
            })
        
        # 选择代价最低的策略
        if not candidates:
            raise RuntimeError("No compatible strategy found!")
        
        best = min(candidates, key=lambda x: x['cost'])
        print(f"[Compiler] Selected strategy: {best['name']} (cost: {best['cost']:.2e})")
        
        return best['compute_fn'], best['name'], best['cost']
    
    def register_custom_strategy(self, strategy: ContractionStrategy, modes: List[str]):
        """
        注册自定义策略
        
        Args:
            strategy: 策略实例
            modes: 要注册到哪些模式，如 ['balanced', 'full']
        """
        self.strategies[strategy.name] = strategy
        
        for mode in modes:
            if mode in self.MODES:
                if strategy.name not in self.MODES[mode]:
                    self.MODES[mode].append(strategy.name)


# ============================================================================
# Original TensorContractor (kept for compatibility)
# ============================================================================

class TensorContractor:
    """
    TensorContractor class for generating optimized tensor contraction expressions.
    
    This class uses opt_einsum to generate contraction expressions but does not
    execute them. The execution is delegated to backend implementations.
    
    Note: This class is kept for compatibility with existing code that uses
    build_with_self_expression, build_core_only_expression, etc.
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
