import jax
import numpy as np
import jax.numpy as jnp
import itertools
import re, random
from typing import Union
from config import Configuration
from copteinsum import ContractorOptEinsum

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
    def generate_example_graph(target=False):
        """Generate an example quantum circuit graph."""
        if target:
            return  "-2-A-5-----C-3-----E-2-\n" \
                    "-2-----B-----------E-2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-----B-6-----D-----2-\n" \
                    "-2-A-3-----C-8-D-----2-"
        else:
            return  "-2-----B-5-C-3-D-----2-\n" \
                    "-2-A-4---------D-----2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-A-3-B-6---------E-2-\n" \
                    "-2---------C-8-----E-2-"
    
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
    def jax_triu_ndindex(n):
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

    def __init__(self, graph):
        """
        Initialize the QCTN with a quantum circuit graph.
        
        Args:
            graph (str): A string representation of the quantum circuit graph.
        """
        self.graph = graph
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.cores = list(set([c for c in graph if c.isupper()]))
        if not self.cores:
            # If no uppercase core symbols found, try to find all chars in the CJK Unified Ideographs range
            self.cores = list(set([c for c in graph if 0x4E00 <= ord(c) <= 0x9FFF]))
        self.ncores = len(self.cores)

        self.cores = sorted(self.cores)

        # This will build the attributes `self.circuit` and `self.adjacency_matrix`
        self._circuit_to_adjacency()

        # Initialize the circuit with input ranks, adjacency matrix, and output ranks
        self.initialize_random_key = jax.random.PRNGKey(0)

        # Initialize the cores with random values
        self.cores_weigts = {}
        self._init_cores()

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
        Convert the quantum circuit graph to an adjacency matrix.
        
        Returns:
            np.ndarray: Adjacency matrix representing the quantum circuit.
        """
        self.adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                self.adjacency_matrix[i, j] = []
        input_ranks = np.empty(self.ncores, dtype=object)
        output_ranks = np.empty(self.ncores, dtype=object)
        for i in range(self.ncores):
            input_ranks[i] = []
            output_ranks[i] = []

        cores = "".join(self.cores)
        dict_core2idx = {core: idx for idx, core in enumerate(self.cores)}
        input_pattern = re.compile(rf"^(\d+)([{cores}])")
        output_pattern = re.compile(rf"([{cores}])(\d+)$")
        connect_pattern = re.compile(rf"([{cores}])(\d+)([{cores}])")

        # print(f"Input Pattern: {input_pattern.pattern}")
        # print(f"Output Pattern: {output_pattern.pattern}")
        # print(f"Connect Pattern: {connect_pattern.pattern}")

        for line in self.qubits:
            line = line.strip().replace("-", "")
            # print(f"Processing line: {line}")
            input_rank, input_core = input_pattern.match(line).groups()
            # print(f"Input Core: {input_core}, Input Rank: {input_rank}")
            output_core, output_rank = output_pattern.search(line).groups()
            input_rank, output_rank = int(input_rank), int(output_rank)
            input_core_idx = dict_core2idx[input_core]
            output_core_idx = dict_core2idx[output_core]
            input_ranks[input_core_idx].append(input_rank)
            output_ranks[output_core_idx].append(output_rank)
            for match in connect_pattern.finditer(line):
                core1, rank1, core2 = match.groups()
                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                self.adjacency_matrix[core1_idx, core2_idx].append(rank1)
                self.adjacency_matrix[core2_idx, core1_idx].append(rank1)

        self.circuit = (input_ranks, self.adjacency_matrix, output_ranks)

    def _init_cores(self):
        """
        Initialize the cores of the quantum circuit with random values.
        
        Returns:
            None: The cores are stored in the `cores_weigts` attribute.
        """

        for idx, core_name in enumerate(self.cores):
            # These ranks should be expressed as lists of integers ordered by the qubits.
            # Therefore we conduct "+" on all of these ranks to obtain the shape of the core.
            input_rank = self.circuit[0][idx]
            output_rank = self.circuit[2][idx]
            adjacency_ranks = self.adjacency_matrix[idx, :]

            core_shape = input_rank + list(itertools.chain.from_iterable(adjacency_ranks)) + output_rank
            core = jax.random.normal(self.initialize_random_key, shape=core_shape) * Configuration.initialize_variance          
            self.cores_weigts[core_name] = core

    def _contract_core_only(self, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network without inputs.
        """

        return engine.contract_core_only(self)

    def _contract_with_inputs(self, inputs: jnp.ndarray = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with given inputs.

        Args:
            inputs (jnp.ndarray): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.

        Returns:
            The result of the contraction operation.
        """

        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if not isinstance(inputs, jnp.ndarray):
            raise TypeError("Inputs must be a jnp.ndarray.")
        if inputs.shape != tuple(itertools.chain.from_iterable(self.circuit[0])):
            raise ValueError(f"Input tensor shape {inputs.shape} does not match expected shape {tuple(itertools.chain.from_iterable(self.circuit[0]))}.")

        return engine.contract_with_inputs(self, inputs)

    def _contract_with_QCTN(self, qctn, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with another QCTN instance.
        
        Args:
            qctn (QCTN): Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation.
        """
        if not isinstance(qctn, QCTN):
            raise TypeError("The argument must be an instance of QCTN.")
        if not list(itertools.chain.from_iterable(self.circuit[0])) == list(itertools.chain.from_iterable(qctn.circuit[0])):
            raise ValueError("Input ranks of the two QCTNs do not match.")
        if not list(itertools.chain.from_iterable(self.circuit[2])) == list(itertools.chain.from_iterable(qctn.circuit[2])):
            raise ValueError("Output ranks of the two QCTNs do not match.")
        
        return engine.contract_with_QCTN(self, qctn)
    
    def _contract_for_core_gradient(self, core_name, inputs=None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network for a specific core gradient.
        
        Args:
            core_name (str): The name of the core to contract for gradient computation.
            inputs (jnp.ndarray, optional): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.
        
        Returns:
            The result of the contraction operation for the specified core.
        """
        if core_name not in self.cores:
            raise ValueError(f"Core '{core_name}' is not part of this QCTN.")
        # Here we would implement the contraction logic using JAX or other libraries.
        # For now, we will raise NotImplementedError as a placeholder.
        
        # Placeholder for contraction logic
        raise NotImplementedError("Contraction logic is not implemented yet.")

    def contract(self, attach: Union[jnp.ndarray, 'QCTN'] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network.

        Args:
            attach (Union[jnp.ndarray, 'QCTN'], optional): The inputs for the contraction operation.
                If a jnp.ndarray is provided, it should be a tensor with the shape matching the input ranks of the circuit.
                If a QCTN instance is provided, it will contract with that instance.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        if attach is None:
            return self._contract_core_only(engine)
        elif isinstance(attach, jnp.ndarray):
            return self._contract_with_inputs(attach, engine)
        elif isinstance(attach, QCTN):
            return self._contract_with_QCTN(attach, engine)
        else:
            raise TypeError("attach must be a jnp.ndarray or an instance of QCTN.")
