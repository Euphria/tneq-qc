from config import Configuration
from tenmul_qc import QCTN, QCTNHelper
from cqctn import ContractorQCTN
from copteinsum import ContractorOptEinsum
import jax, itertools
import jax.numpy as jnp

class UnitTestQCTN:
    """
    Unit test class for QCTN.
    
    This class provides methods to test the functionality of the QCTN class.
    """

    contraction_engine = ContractorOptEinsum if Configuration.contraction_engine == 'opt_einsum' else ContractorQCTN

    @staticmethod
    def test_qctn_initialization(traget=False):
        """Test the initialization of QCTN with a sample graph."""
        example_graph = QCTNHelper.generate_example_graph(traget)
        print(f"Example Graph: \n{example_graph}")
        qctn = QCTN(example_graph)
        print(f"QCTN Adjacency Matrix:\n{qctn.__repr__()}")
        print(f"Cores: {qctn.cores}")
        print(f"Number of Qubits: {qctn.nqubits}")
        print(f"Number of Cores: {qctn.ncores}")
        return qctn

    @staticmethod
    def test_qctn_initialization_with_random_graph():
        """Test the initialization of QCTN with a random graph."""
        example_graph = QCTNHelper.generate_random_example_graph(30, 50)
        print(f"Example Graph: \n{example_graph}")
        qctn = QCTN(example_graph)
        print(f"QCTN Adjacency Matrix:\n{qctn.__repr__()}")
        print(f"Cores: {qctn.cores}")
        print(f"Number of Qubits: {qctn.nqubits}")
        print(f"Number of Cores: {qctn.ncores}")
        return qctn
    
    @staticmethod
    def test_qctn_contract_opt_einsum_core_only():
        """Test the contraction of QCTN using opt_einsum."""
        example_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(example_graph)
        result = qctn._contract_core_only(UnitTestQCTN.contraction_engine)
        print(f"Contraction Result: {result}")
        return result

if __name__ == "__main__":
    # Run unit tests for QCTN

    if False:
        print("Testing QCTN Initialization with Example Graph:")
        qctn_example = UnitTestQCTN.test_qctn_initialization()
        
        print("\nTesting QCTN Initialization with Random Graph:")
        qctn_random = UnitTestQCTN.test_qctn_initialization_with_random_graph()
        
        print("\nTesting QCTN Contraction with opt_einsum:")
        result = UnitTestQCTN.test_qctn_contract_opt_einsum_core_only()
        print(f"Contraction Result: {result}")

        print("\nTesting QCTN Input Contraction with opt_einsum:")
        qctn_example = UnitTestQCTN.test_qctn_initialization()
        inputs = jax.random.normal(jax.random.PRNGKey(0), list(itertools.chain.from_iterable(qctn_example.circuit[0])))
        result = qctn_example.contract(inputs, engine=UnitTestQCTN.contraction_engine)
        print(f"Contraction Result with Inputs: {result}")

        print("\nTesting QCTN Contraction with Another QCTN:")
        qctn_example = UnitTestQCTN.test_qctn_initialization()
        qctn_target = UnitTestQCTN.test_qctn_initialization(traget=True)
        result = qctn_example.contract(qctn_target, engine=UnitTestQCTN.contraction_engine)
        print(f"Contraction Result with Another QCTN: {result}")

    print("\nTesting QCTN Vector Input Contraction with opt_einsum:")
    qctn_example = UnitTestQCTN.test_qctn_initialization()

    inputs = [
        jax.random.normal(jax.random.PRNGKey(i), shape)
        for i, shape in enumerate(itertools.chain.from_iterable(qctn_example.circuit[0]))
    ]
    result = qctn_example.contract(inputs, engine=UnitTestQCTN.contraction_engine)
    print(f"Contraction Result with Inputs: {result}")

    # Further tests can be added here
    print("\nAll tests completed.")
    