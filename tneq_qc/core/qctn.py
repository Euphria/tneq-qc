import numpy as np
import itertools
import re, random
from pathlib import Path
from typing import Union, Any, Optional, Mapping
from ..config import Configuration
from ..backends.copteinsum import ContractorOptEinsum
from .tn_tensor import TNTensor

class QCTNHelper:
    """
    Helper class for Quantum Circuit Tensor Network (QCTN) operations.
    Provides methods for converting quantum circuit graphs to adjacency matrices and counting qubits.
    """

    @staticmethod
    def iter_symbols(extend=False):
        """
        Generate a sequence of symbols for quantum circuit cores.
        If extend is True, use a range of Chinese characters; otherwise, use uppercase letters
        """
 
        if extend:
            symbols = [chr(i) for i in range(0x4E00, 0x9FFF + 1)]
            random.shuffle(symbols)  # Shuffle the symbols for randomness
            symbols = "".join(symbols)
        else:
            symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for symbol in symbols:
            yield symbol

    @staticmethod
    def generate_example_graph(n=16, target=False):
        """Generate an example quantum circuit graph."""
        if target:
            return  "-2-A-5-----C-3-----E-2-\n" \
                    "-2-----B----4------E-2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-----B-6-----D-----2-\n" \
                    "-2-A-3-----C-8-D-----2-"
        else:
            def generate_std_graph(n):
                graph = ""
                # char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                # char_list = [next(QCTNHelper.iter_symbols(True)) for _ in range(n)]
                import opt_einsum
                char_list = [opt_einsum.get_symbol(i) for i in range(n)]

                dim_char = '3'

                for i in range(n):
                    cid = i - 1
                    nid = i
                    if i == 0:
                        line = f"-{dim_char}-" + char_list[i] + (n - 2) * 6 * "-" + f"-{dim_char}-"
                    elif i == n - 1:
                        line = f"-{dim_char}-" + (n - 2) * 6 * "-" + char_list[cid] + f"-{dim_char}-"
                    else:
                        line = f"-{dim_char}-"
                        line += cid * 6 * "-"
                        line += char_list[cid]
                        line += f"--{dim_char}--"
                        line += char_list[nid]
                        line += (n - nid - 2) * 6 * "-"
                        line += f"-{dim_char}-"
                    
                    graph += line + "\n"
                return graph
            
            return generate_std_graph(n)
        
            # return  "-3-A-3-"
            # return  "-3-A-3-B-3-C-3-D-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-"

            # return  "-3-A-3-B-3-C-3-\n" \
            #         "-3-A-3-B-3-C-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-"

            # return  "-3-A-----3-\n" \
            #         "-3-A-3-B-3-\n" \
            #         "-3-----B-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-"


            # return  "-3-a-3-b-3-c---------3-\n" \
            #         "-3-a-3-b-------------3-\n" \
            #         "-3-a-3-b-3-c---------3-\n" \
            #         "-3---------c---------3-"
        
            # return  "-3-a-3-----c-3-d-3-e-3-\n" \
            #         "-3-a-3-b-3-----d-3-e-3-\n" \
            #         "-3-a-3-b-3-c-3-----e-3-\n" \
            #         "-3-----b-3-c-3-d-3-e-3-"
        
            # return  "-3-A--------3--C--3--D-3-\n" \
            #         "-3-A--3--B--------3--D-3-\n" \
            #         "-3-------B--3--C--3--D-3-"
        
            # return  "-3-a-------------3-e-3-\n" \
            #         "-3-a-3-b-----3-d-3-e-3-\n" \
            #         "-3-----b-3-c-3-d-----3-\n" \
            #         "-3---------c---------3-"
            
            return  "-3-a-------------3-e-3-\n" \
                    "-3-a-3-b-----3-d-3-e-3-\n" \
                    "-3-f-3-b-3-c-3-d-3-g-3-\n" \
                    "-3-f-3-----c-----3-g-3-"
        
            

            # return  "-3-A-------------------3-\n" \
            #         "-3-A--3--B-------------3-\n" \
            #         "-3-------B--3--C-------3-\n" \
            #         "-3-------------C--3--D-3-\n" \
            #         "-3-------------------D-3-"
        
            return  "-3-A-------------------------3-\n" \
                    "-3-A--3--B-------------------3-\n" \
                    "-3-------B--3--C-------------3-\n" \
                    "-3-------------C--3--D-------3-\n" \
                    "-3-------------------D--3--E-3-\n" \
                    "-3-------------------------E-3-"
            

            # circuit_states.
            # (1, K), 001 


            # return  "-2-A-2-"
            # return  "-2-A-3-B-4-C-3-D-2-"
        
            # return  "-2-A-2-\n" \
            #         "-2-A-2-"

            return  "-2-A-------------------4-\n" \
                    "-2-A--3--B-------------4-\n" \
                    "-2-------B--3--C-------4-\n" \
                    "-2-------------C--3--D-4-\n" \
                    "-2-------------------D-4-"


            return  "-2-A-5-----C-3-----E-2-\n" \
                    "-2-----B----4------E-2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-----B-6-----D-----2-\n" \
                    "-2-A-3-----C-8-D-----2-"
        
            return  "-2-A--------5--C--5--D-5-\n" \
                    "-2-A--5--B--------5--D-5-\n" \
                    "-2-------B--5--C--5--D-5-"
        
            return  "-2-A--------2--C--2--D-2-\n" \
                    "-2-A--2--B--------2--D-2-\n" \
                    "-2-------B--2--C--2--D-2-"
        

            return  "-2-A--------5--C--7--D--9--E--9--F--------9-\n" \
                    "-2-A--3--B--------7--D--------9--F--7--G-9-\n" \
                    "-2-------B--5--C--7--D--9--E--------9--G-9-"
        

            return  "-2-A--------5--C--------9--E--9--F--7-----9--H-9-\n" \
                    "-2-A--3--B--5--C--7--D--9--E--9--F--7--G--9--H-9-\n" \
                    "-2-------B--------7--D--9--E--9--------G-------9-"


            return  "-2-A--------5--C-------7-\n" \
                    "-2-A--3--B--5--C--7--D-9-\n" \
                    "-2-------B--------7--D-9-"
        
            return  "-2-A--------5--C-------5-\n" \
                    "-2-A--5--B--5--C--5--D-5-\n" \
                    "-2-------B--------5--D-5-"

        
            

            # return  "-2-A-----------------------------------------A-2-\n" \
            #         "-2-A--2--B-----------------------------B--2--A-2-\n" \
            #         "-2-------B--2--C-----------------C--2--B-------2-\n" \
            #         "-2-------------C--2--D-----D--2--C-------------2-\n" \
            #         "-2-------------------D--2--D-------------------2-"

            return  "-2-----B-5-C-3-D-----2-\n" \
                    "-2-A-4---------D-----2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-A-3-B-6---------E-2-\n" \
                    "-2---------C-8-----E-2-"
            """
            -2-A-------------------2-
            -2-A--2--B-------------2-
            -2-------B--2--C-------2-
            -2-------------C--2--D-2-
            -2-------------------D-2-
            """


    @staticmethod
    def generate_example_graph2():
        """Generate an example quantum circuit graph."""

    
    @staticmethod
    def generate_random_example_graph(nqubits=5, ncores=3):
        """Generate a random quantum circuit graph with specified number of qubits and cores."""

        cores = "".join([next(QCTNHelper.iter_symbols(True)) for _ in range(ncores)])
        graph = ""
        for i in range(nqubits):
            qubit = f"-{np.random.randint(2, 10)}-"
            for j in cores:
                if np.random.rand() > 0.5:
                    qubit += f"{j}-{np.random.randint(2, 10)}-"

            graph += f"{qubit}\n"

        return graph.strip()
    
    @staticmethod
    def triu_ndindex(n):
        """Generate indices for the upper triangular part of a square matrix."""
        for i in range(n):
            for j in range(i + 1, n):
                yield (i, j)

class QCTN:
    """
    Quantum Circuit Tensor Network (QCTN) class for quantum circuit simulation.
    
    Initialization Format:
        - A graph representing the quantum circuit, where open edges are qubits and marks are cores.
        - Each core is a tensor with a shape corresponding to the number of qubits it connects to.

    Example:
        -2-----B-5-C-3-D-----2-
        -2-A-4---------D-----2-
        -2-A-4-B-7-C-2-D-4-E-2-
        -2-A-3-B-6---------E-2-
        -2---------C-8-----E-2-

        where:
            - A, B, C, D, E are cores (tensors).
            - The numbers represent the rank of each core.

    Attributes:
        nqubits (int): Number of qubits in the quantum circuit.
        adjacency_matrix: (np.ndarray): Adjacency matrix representing the connection ranks with empty diagonal entries.
        circuit (tuple): (Input ranks, Connection ranks, Output ranks) for each core.
 
    """

    def __init__(self, graph, backend=None):
        """
        Initialize the QCTN with a quantum circuit graph.
        
        Args:
            graph (str): A string representation of the quantum circuit graph.
            backend (ComputeBackend): The backend to use for computation.
        """
        self.graph = graph
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.qubit_indices = list(range(self.nqubits))
        
        import opt_einsum
        full_cores = set([opt_einsum.get_symbol(i) for i in range(10000)])
        # full_cores = set([next(QCTNHelper.iter_symbols(True)) for _ in range(20000)])

        self.cores = list(set([c for c in graph if c in full_cores]))
        print(f"num cores: {len(self.cores)}")


        # self.cores = [opt_einsum.get_symbol(i) for i in range(self.nqubits-1)]


        # self.cores = list(set([c for c in graph if c.isupper()]))
        # if not self.cores:
        #     # If no uppercase core symbols found, try to find all chars in the CJK Unified Ideographs range
        #     self.cores = list(set([c for c in graph if 0x4E00 <= ord(c) <= 0x9FFF]))
        self.ncores = len(self.cores)

        self.cores = sorted(self.cores)

        # This will build the attributes `self.circuit` and `self.adjacency_matrix`
        self._circuit_to_adjacency()

        self.backend = backend
        self._loaded_metadata: Optional[Mapping[str, str]] = None

        # Initialize the circuit with input ranks, adjacency matrix, and output ranks
        # self.initialize_random_key = jax.random.PRNGKey(0)

        # Initialize the cores with random values
        self.cores_weights = {}
        self._init_cores()

        # Placeholders for einsum expressions
        self.einsum_expr = None

    @classmethod
    def envolve_from_another_qctn(cls, qctn, strategies=None):
        """
        Create a new QCTN instance by evolving from another QCTN instance.
        
        Args:
            qctn (QCTN): The original QCTN instance to evolve from.
            strategies (list, optional): A list of strategies for evolution. Defaults to None.
        
        Returns:
            QCTN: A new QCTN instance evolved from the original.
        """
        if strategies is None \
           or (isinstance(strategies, list) and not strategies):
            # If no strategies are provided, simply copy the original QCTN
            # This is useful for cases where we want to create a new instance without any modifications.
            # The optimization can be skipped.
            if isinstance(qctn, cls):
                # If qctn is already an instance of QCTN, copy it
                return cls.copy(qctn)
            else:
                # If qctn is not an instance of QCTN, raise an error
                raise TypeError("qctn must be an instance of QCTN.")

        if isinstance(strategies, function):
            new_graph = strategies(qctn.graph)
            return cls(new_graph)
        elif isinstance(strategies, list):
            # Apply each strategy to the original graph
            new_graph = qctn.graph
            for strategy in strategies:
                if callable(strategy):
                    new_graph = strategy(new_graph)
                else:
                    raise TypeError("Each strategy must be a callable function.")
            return cls(new_graph)

    def __repr__(self):
        """
        String representation of the QCTN object.
        """
        adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                adjacency_matrix[i, j] = str(self.adjacency_matrix[i, j])
        
        circuit_input = [str(rank) for rank in self.circuit[0]]
        circuit_output = [str(rank) for rank in self.circuit[2]]

        return f"circuit_input = \n{circuit_input}\n adjacency_matrix = \n{adjacency_matrix}\n circuit_output = \n{circuit_output}\n"

    def _circuit_to_adjacency(self,):
        """
        Convert the quantum circuit graph to adjacency table.
        
        This method builds self.adjacency_table, a list where each element corresponds to a core
        and contains a dict with:
        - 'core_idx': int, index of the core
        - 'core_name': str, name of the core
        - 'in_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For input edges (from circuit input), neighbor_idx = -1, neighbor_name = ""
        - 'out_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For output edges (to circuit output), neighbor_idx = -1, neighbor_name = ""
        - 'input_shape': list of input ranks (from in_edge_list)
        - 'output_shape': list of output ranks (from out_edge_list)
        - 'input_dim': int, product of input_shape
        - 'output_dim': int, product of output_shape
        """
        
        cores = "".join(self.cores)
        dict_core2idx = {core: idx for idx, core in enumerate(self.cores)}
        self.dict_core2idx = dict_core2idx  # Store for later use
        
        # Initialize adjacency_table
        self.adjacency_table = []
        for idx, core_name in enumerate(self.cores):
            self.adjacency_table.append({
                'core_idx': idx,
                'core_name': core_name,
                'in_edge_list': [],
                'out_edge_list': [],
                'input_shape': [],
                'output_shape': [],
                'input_dim': 1,
                'output_dim': 1,
            })
        
        input_pattern = re.compile(rf"^(\d+)([{cores}])")
        output_pattern = re.compile(rf"([{cores}])(\d+)$")
        connect_pattern = re.compile(rf"([{cores}])(\d+)(?=[{cores}])")

        for qubit_idx, line in enumerate(self.qubits):
            # print(f'qubit_idx: {qubit_idx}, line: {len(line)}, {line[-10:]}')
            # if qubit_idx == 2000:
            #     print(line)
            line = line.strip().replace("-", "")
            input_rank, input_core = input_pattern.match(line).groups()
            output_core, output_rank = output_pattern.search(line).groups()
            input_rank, output_rank = int(input_rank), int(output_rank)
            input_core_idx = dict_core2idx[input_core]
            output_core_idx = dict_core2idx[output_core]
            
            # Add input edge: from circuit input (-1, "") to input_core
            self.adjacency_table[input_core_idx]['in_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': input_rank,
                'qubit_idx': qubit_idx
            })
            
            # Add output edge: from output_core to circuit output (-1, "")
            self.adjacency_table[output_core_idx]['out_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': output_rank,
                'qubit_idx': qubit_idx
            })
            
            for match in connect_pattern.finditer(line):
                end_pos = match.end()
                if end_pos >= len(line):
                    print(f"Warning: end_pos {end_pos} out of range for line '{line}'")
                    break

                core1, rank1 = match.groups()
                core2 = line[end_pos]

                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                
                # Add to adjacency table
                # core1 -> core2: out_edge for core1, in_edge for core2
                self.adjacency_table[core1_idx]['out_edge_list'].append({
                    'neighbor_idx': core2_idx,
                    'neighbor_name': core2,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })
                self.adjacency_table[core2_idx]['in_edge_list'].append({
                    'neighbor_idx': core1_idx,
                    'neighbor_name': core1,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })

        # Compute input_shape, output_shape, input_dim, output_dim for each core
        for core_info in self.adjacency_table:
            core_info['input_shape'] = [edge['edge_rank'] for edge in core_info['in_edge_list']]
            core_info['output_shape'] = [edge['edge_rank'] for edge in core_info['out_edge_list']]
            core_info['input_dim'] = int(np.prod(core_info['input_shape'])) if core_info['input_shape'] else 1
            core_info['output_dim'] = int(np.prod(core_info['output_shape'])) if core_info['output_shape'] else 1

        # Build adjacency_matrix from adjacency_table for backward compatibility
        self.adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                self.adjacency_matrix[i, j] = []
        
        for core_info in self.adjacency_table:
            core_idx = core_info['core_idx']
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] >= 0:  # Skip output edges (to circuit output)
                    self.adjacency_matrix[core_idx, edge['neighbor_idx']].append(edge['edge_rank'])
                    self.adjacency_matrix[edge['neighbor_idx'], core_idx].append(edge['edge_rank'])

        # Build circuit tuple for backward compatibility
        input_ranks = np.empty(self.ncores, dtype=object)
        output_ranks = np.empty(self.ncores, dtype=object)
        for i in range(self.ncores):
            input_ranks[i] = self.adjacency_table[i]['input_shape'].copy()
            output_ranks[i] = self.adjacency_table[i]['output_shape'].copy()
        self.circuit = (input_ranks, self.adjacency_matrix, output_ranks)

        # for debug, print adjacency_table
        # for core_info in self.adjacency_table:
        #     print(f"Core {core_info['core_name']} (idx {core_info['core_idx']}):")
        #     print(f"  input_shape: {core_info['input_shape']}, output_shape: {core_info['output_shape']}")
        #     print(f"  input_dim: {core_info['input_dim']}, output_dim: {core_info['output_dim']}")
        #     print(f"  In edges: {core_info['in_edge_list']}")
        #     print(f"  Out edges: {core_info['out_edge_list']}")

    def _init_cores(self):
        """
        Initialize the cores of the quantum circuit with random values.
        
        For each core, use the pre-computed values from adjacency_table:
        - input_shape: ranks from in_edge_list (already ordered by qubit_idx)
        - output_shape: ranks from out_edge_list (already ordered by qubit_idx)
        - input_dim: product of input_shape
        - output_dim: product of output_shape
        
        The core tensor is initialized with shape [input_dim, output_dim], 
        then reshaped to input_shape + output_shape.
        
        Returns:
            None: The cores are stored in the `cores_weights` attribute.
        """

        for idx, core_info in enumerate(self.adjacency_table):
            core_name = core_info['core_name']
            input_shape = core_info['input_shape']
            output_shape = core_info['output_shape']
            input_dim = core_info['input_dim']
            output_dim = core_info['output_dim']
            
            # print(f"_init_cores: {idx} {input_shape} {output_shape} {input_dim} {output_dim}")

            # Initialize core with shape [input_dim, output_dim]
            core = self.backend.init_random_core([input_dim, output_dim])
            
            # Reshape to input_shape + output_shape
            full_shape = input_shape + output_shape
            core = self.backend.reshape(core, full_shape)

            self.cores_weights[core_name] = TNTensor(core)
            self.cores_weights[core_name].auto_scale()

    def save_cores(self, file_path: Union[str, Path], metadata: Optional[Mapping[str, str]] = None):
        """Save all core tensors into a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before saving cores.")

        try:
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise ImportError("safetensors is required to save cores; install it with `pip install safetensors`.") from exc

        tensor_dict = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                tensor_dict[f"core_{core_name}"] = self.backend.tensor_to_numpy(tensor.tensor * tensor.scale)
            else:
                tensor_dict[f"core_{core_name}"] = self.backend.tensor_to_numpy(tensor)

        metadata_dict = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}
        save_file(tensor_dict, str(file_path), metadata=metadata_dict)

    def load_cores(self, file_path: Union[str, Path], strict: bool = True) -> Mapping[str, str]:
        """Load saved core tensors from a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before loading cores.")

        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load cores; install it with `pip install safetensors`.") from exc

        result = load_file(str(file_path))
        if isinstance(result, tuple) and len(result) == 2:
            tensor_dict, metadata = result
        else:
            tensor_dict = result
            metadata = {}

        for core_name in self.cores:
            key = f"core_{core_name}"
            if key not in tensor_dict:
                if strict:
                    raise KeyError(f"Missing tensor for core {core_name} in {file_path}")
                continue
            array = tensor_dict[key]
            tensor = self.backend.convert_to_tensor(array)
            tn_tensor = TNTensor(tensor)
            tn_tensor.auto_scale()
            self.cores_weights[core_name] = tn_tensor

        metadata_dict = {str(k): str(v) for k, v in metadata.items()}
        self._loaded_metadata = metadata_dict
        return metadata_dict

    @classmethod
    def from_pretrained(
        cls,
        graph: str,
        file_path: Union[str, Path],
        backend=None,
        strict: bool = True,
    ) -> "QCTN":
        """Create a QCTN instance loading core tensors from safetensors."""

        if backend is None:
            from ..backends.backend_factory import BackendFactory

            backend = BackendFactory.get_default_backend()

        instance = cls(graph, backend=backend)
        instance.load_cores(file_path, strict=strict)
        return instance


    def _contract_core_only(self, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network without inputs.
        """

        return engine.contract_core_only(self)

    def _contract_with_inputs(self, inputs: Any = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with given inputs.

        Args:
            inputs (Any): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.

        Returns:
            The result of the contraction operation.
        """

        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if not isinstance(inputs, self.backend.get_tensor_type()):
            raise TypeError(f"Inputs must be a {self.backend.get_tensor_type()}.")
        if inputs.shape != tuple(itertools.chain.from_iterable(self.circuit[0])):
            raise ValueError(f"Input tensor shape {inputs.shape} does not match expected shape {tuple(itertools.chain.from_iterable(self.circuit[0]))}.")

        return engine.contract_with_inputs(self, inputs)

    def _contract_with_vector_inputs(self, inputs: list = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with given inputs.

        Args:
            inputs (list): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.

        Returns:
            The result of the contraction operation.
        """

        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if not all(isinstance(t, self.backend.get_tensor_type()) for t in inputs):
            raise TypeError(f"All elements in the list must be {self.backend.get_tensor_type()}.")
        if len(inputs) != self.nqubits:
            raise ValueError(f"Expected {self.nqubits} input vectors, got {len(inputs)}.")
        
        # Check dimensions of all inputed vectors
        if not all(t.ndim == 1 for t in inputs):
            raise ValueError("All input tensors must be 1-dimensional vectors.")

        if tuple(t.shape[0] for t in inputs) != tuple(itertools.chain.from_iterable(self.circuit[0])):
            raise ValueError(f"Input tensor shapes {tuple(t.shape for t in inputs)} do not match expected shape {tuple(itertools.chain.from_iterable(self.circuit[0]))}.")

        return engine.contract_with_vector_inputs(self, inputs)

    def _contract_with_QCTN(self, qctn, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with another QCTN instance.
        
        Args:
            qctn (QCTN): Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation.
        """

        if not list(itertools.chain.from_iterable(self.circuit[0])) == list(itertools.chain.from_iterable(qctn.circuit[0])):
            raise ValueError("Input ranks of the two QCTNs do not match.")
        if not list(itertools.chain.from_iterable(self.circuit[2])) == list(itertools.chain.from_iterable(qctn.circuit[2])):
            raise ValueError("Output ranks of the two QCTNs do not match.")
        
        return engine.contract_with_QCTN(self, qctn)
    
    def _contract_with_QCTN_for_gradient(self, qctn, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network for a specific core gradient.
        
        Args:
            qctn : Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation for core gradient.
        """

        if not list(itertools.chain.from_iterable(self.circuit[0])) == list(itertools.chain.from_iterable(qctn.circuit[0])):
            raise ValueError("Input ranks of the two QCTNs do not match.")
        if not list(itertools.chain.from_iterable(self.circuit[2])) == list(itertools.chain.from_iterable(qctn.circuit[2])):
            raise ValueError("Output ranks of the two QCTNs do not match.")

        return engine.contract_with_QCTN_for_gradient(self, qctn)

    def contract(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network.

        Args:
            attach (Union[Any, list[Any], 'QCTN'], optional): The inputs for the contraction operation.
                If a tensor is provided, it should be a tensor with the shape matching the input ranks of the circuit.
                If a list of tensors is provided, it should contain vectors for each qubit.
                If a QCTN instance is provided, it will contract with that instance.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        if attach is None:
            return self._contract_core_only(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            print('contract with tensor')
            return self._contract_with_inputs(attach, engine)
        elif isinstance(attach, list):
            print('contract with list of tensors')
            return self._contract_with_vector_inputs(attach, engine)
        elif isinstance(attach, QCTN):
            print('contract with QCTN')
            return self._contract_with_QCTN(attach, engine)
        else:
            raise TypeError(f"attach must be a {self.backend.get_tensor_type()}, a list of {self.backend.get_tensor_type()} or an instance of QCTN.")
    
    def _contract_with_self(self, engine=ContractorOptEinsum, circuit_array_input=None, circuit_list_input=None):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        return engine.contract_with_self(self, circuit_array_input, circuit_list_input)

    def _contract_with_self_for_gradient(self, engine=ContractorOptEinsum, circuit_array_input=None, circuit_list_input=None):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        return engine.contract_with_self_for_gradient(self, circuit_array_input, circuit_list_input)

    def contract_with_self(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """
        if attach is None:
            return self._contract_with_self(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            return self._contract_with_self(engine, circuit_array_input=attach)
        elif isinstance(attach, list):
            raise TypeError("attach must be None when contracting with self.")
        elif isinstance(attach, QCTN):
            raise TypeError("attach must be None when contracting with self.")
        else:
            raise TypeError("attach must be None when contracting with self.")
    
    def contract_with_self_for_gradient(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """
        if attach is None:
            return self._contract_with_self_for_gradient(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            return self._contract_with_self_for_gradient(engine, circuit_array_input=attach)
        elif isinstance(attach, list):
            raise TypeError("attach must be None when contracting with self.")
        elif isinstance(attach, QCTN):
            raise TypeError("attach must be None when contracting with self.")
        else:
            raise TypeError("attach must be None when contracting with self.")

    def contract_with_QCTN_for_gradient(self, attach, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network, return the loss and the gradient for all cores,
        The attach must be a QCTN instance.
        The loss for gradient computation is (X @ Y^T - 1) ** 2, where X is self and Y is the attach QCTN.
        The gradient only computes the core gradients, not the input gradients.
        

        Args:
            attach ('QCTN'): The inputs for the contraction operation.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of value and grad.
        """
        if not isinstance(attach, QCTN):
            raise TypeError("attach must be an instance of QCTN.")
        return self._contract_with_QCTN_for_gradient(attach, engine)
    
    def optimize_contract_with_QCTN(self, attach, optimizer, engine=ContractorOptEinsum):
        """
        Optimize the contraction with another QCTN instance using a specified optimizer.

        Args:
            attach (QCTN): The QCTN instance to contract with.
            optimizer (Optimizer): The optimizer to use for the optimization process.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The optimized parameters after the contraction operation.
        """
        if not isinstance(attach, QCTN):
            raise TypeError("attach must be an instance of QCTN.")
        
        return optimizer.optimize(self.contract_with_QCTN_for_gradient, attach, engine=engine)
    
