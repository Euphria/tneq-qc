"""
Unit tests for the refactored contract and backend architecture.

Tests cover:
1. BackendFactory - backend creation and management
2. TensorContractor - einsum expression generation
3. Engine - contraction execution with different backends
4. Backend comparison - JAX vs PyTorch equivalence
"""

import unittest
import numpy as np
from tneq_qc.backends import BackendFactory, JAXBackend, PyTorchBackend
from tneq_qc.core import TensorContractor, Engine, QCTN


class TestBackendFactory(unittest.TestCase):
    """Test BackendFactory functionality."""

    def test_create_jax_backend(self):
        """Test creating JAX backend."""
        backend = BackendFactory.create_backend('jax')
        self.assertIsInstance(backend, JAXBackend)
        self.assertEqual(backend.get_backend_name(), 'jax')

    def test_create_pytorch_backend(self):
        """Test creating PyTorch backend if available."""
        try:
            backend = BackendFactory.create_backend('pytorch')
            self.assertIsInstance(backend, PyTorchBackend)
            self.assertEqual(backend.get_backend_name(), 'pytorch')
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_invalid_backend(self):
        """Test that invalid backend name raises ValueError."""
        with self.assertRaises(ValueError):
            BackendFactory.create_backend('invalid_backend')

    def test_set_default_backend(self):
        """Test setting and getting default backend."""
        BackendFactory.set_default_backend('jax')
        default_backend = BackendFactory.get_default_backend()
        self.assertEqual(default_backend.get_backend_name(), 'jax')

    def test_get_default_backend_creates_jax_if_none(self):
        """Test that get_default_backend creates JAX backend if not set."""
        # Reset the default backend
        BackendFactory._backend_instance = None
        default_backend = BackendFactory.get_default_backend()
        self.assertEqual(default_backend.get_backend_name(), 'jax')


class TestTensorContractor(unittest.TestCase):
    """Test TensorContractor expression generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph_string = (
            "-2-A-2-\n"
            "-2-A-2-\n"
            "-2-A-2-"
        )
        self.qctn = QCTN(self.graph_string)
        self.contractor = TensorContractor()

    def test_build_core_only_expression(self):
        """Test generating core-only contraction expression."""
        einsum_eq, shapes = self.contractor.build_core_only_expression(self.qctn)
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        self.assertIn('->', einsum_eq)
        self.assertEqual(len(shapes), len(self.qctn.cores))

    def test_build_with_inputs_expression(self):
        """Test generating with-inputs contraction expression."""
        input_shape = (2, 2, 2)
        einsum_eq, shapes = self.contractor.build_with_inputs_expression(
            self.qctn, input_shape
        )
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        self.assertIn('->', einsum_eq)
        self.assertEqual(len(shapes), len(self.qctn.cores) + 1)  # +1 for input
        self.assertEqual(shapes[0], input_shape)

    def test_build_with_vector_inputs_expression(self):
        """Test generating with-vector-inputs contraction expression."""
        inputs_shapes = [(2,), (2,), (2,)]
        einsum_eq, shapes = self.contractor.build_with_vector_inputs_expression(
            self.qctn, inputs_shapes
        )
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        self.assertIn('->', einsum_eq)
        self.assertEqual(len(shapes), len(self.qctn.cores) + len(inputs_shapes))

    def test_build_with_qctn_expression(self):
        """Test generating with-QCTN contraction expression."""
        target_qctn = QCTN(self.graph_string)
        einsum_eq, shapes = self.contractor.build_with_qctn_expression(
            self.qctn, target_qctn
        )
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        self.assertIn('->', einsum_eq)
        self.assertEqual(len(shapes), len(self.qctn.cores) + len(target_qctn.cores))

    def test_build_with_self_expression(self):
        """Test generating with-self contraction expression."""
        einsum_eq, shapes = self.contractor.build_with_self_expression(self.qctn)
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        self.assertIn('->', einsum_eq)
        # Should have cores + reversed cores
        self.assertEqual(len(shapes), 2 * len(self.qctn.cores))

    def test_build_with_self_expression_with_input(self):
        """Test generating with-self contraction expression with input."""
        input_shape = (2, 2, 2)
        einsum_eq, shapes = self.contractor.build_with_self_expression(
            self.qctn, input_shape
        )
        
        self.assertIsInstance(einsum_eq, str)
        self.assertIsInstance(shapes, list)
        # Should have input + cores + reversed cores + input
        self.assertEqual(len(shapes), 2 * len(self.qctn.cores) + 2)

    def test_create_contract_expression(self):
        """Test creating optimized contract expression."""
        einsum_eq, shapes = self.contractor.build_core_only_expression(self.qctn)
        expr = self.contractor.create_contract_expression(einsum_eq, shapes)
        
        # Check that expression is callable
        self.assertTrue(callable(expr))


class TestEngineJAX(unittest.TestCase):
    """Test Engine with JAX backend."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph_string = (
            "-2-A-2-\n"
            "-2-A-2-\n"
            "-2-A-2-"
        )
        self.qctn = QCTN(self.graph_string)
        self.engine = Engine(backend='jax')

    def test_executor_initialization(self):
        """Test executor initialization."""
        self.assertIsNotNone(self.engine.backend)
        self.assertEqual(self.engine.backend.get_backend_name(), 'jax')
        self.assertIsNotNone(self.engine.contractor)

    def test_contract_core_only(self):
        """Test core-only contraction."""
        result = self.engine.contract_core_only(self.qctn)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))
        # Result should be a tensor with some shape
        self.assertGreater(len(result.shape), 0)

    def test_contract_with_inputs(self):
        """Test contraction with inputs."""
        inputs = np.random.rand(2, 2, 2).astype(np.float32)
        result = self.engine.contract_with_inputs(self.qctn, inputs)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_vector_inputs(self):
        """Test contraction with vector inputs."""
        inputs = [
            np.random.rand(2).astype(np.float32),
            np.random.rand(2).astype(np.float32),
            np.random.rand(2).astype(np.float32)
        ]
        result = self.engine.contract_with_vector_inputs(self.qctn, inputs)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_qctn(self):
        """Test contraction with another QCTN."""
        target_qctn = QCTN(self.graph_string)
        result = self.engine.contract_with_qctn(self.qctn, target_qctn)
        
        self.assertIsNotNone(result)
        # Result should be a scalar for complete contraction
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_self(self):
        """Test contraction with self."""
        result = self.engine.contract_with_self(self.qctn)
        
        self.assertIsNotNone(result)
        # Result should be a scalar
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_self_with_input(self):
        """Test contraction with self and input."""
        inputs = np.random.rand(2, 2, 2).astype(np.float32)
        result = self.engine.contract_with_self(self.qctn, inputs)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_self_for_gradient(self):
        """Test contraction with self and gradient computation."""
        loss, grads = self.engine.contract_with_self_for_gradient(self.qctn)
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(grads, (list, tuple))
        self.assertEqual(len(grads), len(self.qctn.cores))
        
        # Check that gradients have correct shapes
        for i, core_name in enumerate(self.qctn.cores):
            grad = grads[i]
            core_shape = self.qctn.cores_weights[core_name].shape
            self.assertEqual(grad.shape, core_shape)

    def test_contract_with_self_for_gradient_with_input(self):
        """Test contraction with self and gradient computation with input."""
        inputs = np.random.rand(2, 2, 2).astype(np.float32)
        loss, grads = self.engine.contract_with_self_for_gradient(self.qctn, inputs)
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(grads, (list, tuple))
        self.assertEqual(len(grads), len(self.qctn.cores))

    def test_contract_with_qctn_for_gradient(self):
        """Test contraction with QCTN and gradient computation."""
        target_qctn = QCTN(self.graph_string)
        loss, grads = self.engine.contract_with_qctn_for_gradient(
            self.qctn, target_qctn
        )
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(grads, (list, tuple))
        self.assertEqual(len(grads), len(self.qctn.cores))

    def test_expression_caching(self):
        """Test that expressions are cached after first use."""
        # First call
        result1 = self.engine.contract_core_only(self.qctn)
        self.assertTrue(hasattr(self.qctn, '_contract_expr_core_only'))
        
        # Second call should use cached expression
        result2 = self.engine.contract_core_only(self.qctn)
        self.assertIsNotNone(result2)


class TestEnginePyTorch(unittest.TestCase):
    """Test Engine with PyTorch backend."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
        
        if self.torch_available:
            self.graph_string = (
                "-2-A-2-\n"
                "-2-A-2-\n"
                "-2-A-2-"
            )
            self.qctn = QCTN(self.graph_string)
            self.engine = Engine(backend='pytorch')

    def test_executor_initialization(self):
        """Test executor initialization with PyTorch."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        self.assertIsNotNone(self.engine.backend)
        self.assertEqual(self.engine.backend.get_backend_name(), 'pytorch')

    def test_contract_core_only(self):
        """Test core-only contraction with PyTorch."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        result = self.engine.contract_core_only(self.qctn)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))

    def test_contract_with_inputs(self):
        """Test contraction with inputs using PyTorch."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        inputs = np.random.rand(2, 2, 2).astype(np.float32)
        result = self.engine.contract_with_inputs(self.qctn, inputs)
        self.assertIsNotNone(result)

    def test_contract_with_self(self):
        """Test contraction with self using PyTorch."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        result = self.engine.contract_with_self(self.qctn)
        self.assertIsNotNone(result)


class TestBackendComparison(unittest.TestCase):
    """Test equivalence between JAX and PyTorch backends."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
        
        self.graph_string = (
            "-2-A-2-\n"
            "-2-A-2-"
        )

    def test_core_only_equivalence(self):
        """Test that JAX and PyTorch give similar results for core-only."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        # Create separate QCTN instances with same structure
        qctn_jax = QCTN(self.graph_string)
        qctn_pytorch = QCTN(self.graph_string)
        
        # Copy weights to ensure they're identical
        for core in qctn_jax.cores:
            qctn_pytorch.cores_weights[core] = qctn_jax.cores_weights[core].copy()
        
        # Execute with both backends
        jax_engine = Engine(backend='jax')
        pytorch_engine = Engine(backend='pytorch')
        
        jax_result = jax_engine.contract_core_only(qctn_jax)
        pytorch_result = pytorch_engine.contract_core_only(qctn_pytorch)
        
        # Convert to numpy for comparison
        jax_result_np = np.array(jax_result)
        pytorch_result_np = np.array(pytorch_result)
        
        # Check that results are close
        np.testing.assert_allclose(
            jax_result_np, 
            pytorch_result_np, 
            rtol=1e-5, 
            atol=1e-6
        )

    def test_with_inputs_equivalence(self):
        """Test that JAX and PyTorch give similar results with inputs."""
        if not self.torch_available:
            self.skipTest("PyTorch not installed")
        
        # Create separate QCTN instances
        qctn_jax = QCTN(self.graph_string)
        qctn_pytorch = QCTN(self.graph_string)
        
        # Copy weights
        for core in qctn_jax.cores:
            qctn_pytorch.cores_weights[core] = qctn_jax.cores_weights[core].copy()
        
        # Same inputs
        inputs = np.random.rand(2, 2).astype(np.float32)
        
        # Execute with both backends
        jax_engine = Engine(backend='jax')
        pytorch_engine = Engine(backend='pytorch')
        
        jax_result = jax_engine.contract_with_inputs(qctn_jax, inputs)
        pytorch_result = pytorch_engine.contract_with_inputs(qctn_pytorch, inputs)
        
        # Convert to numpy for comparison
        jax_result_np = np.array(jax_result)
        pytorch_result_np = np.array(pytorch_result)
        
        # Check that results are close
        np.testing.assert_allclose(
            jax_result_np, 
            pytorch_result_np, 
            rtol=1e-5, 
            atol=1e-6
        )


class TestEngineDefaultBackend(unittest.TestCase):
    """Test Engine with default backend."""

    def test_default_backend_is_jax(self):
        """Test that default backend is JAX."""
        engine = Engine()
        self.assertEqual(engine.backend.get_backend_name(), 'jax')

    def test_contract_with_default_backend(self):
        """Test contraction with default backend."""
        graph_string = "-2-A-2-\n-2-A-2-"
        qctn = QCTN(graph_string)
        
        engine = Engine()
        result = engine.contract_core_only(qctn)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'shape'))


class TestEngineMultipleQCTN(unittest.TestCase):
    """Test engine with multiple QCTN instances."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = Engine(backend='jax')

    def test_different_qctn_structures(self):
        """Test engine with different QCTN structures."""
        graph1 = "-2-A-2-\n-2-A-2-"
        graph2 = "-2-B-3-C-2-\n-2-B-2-C-2-"
        
        qctn1 = QCTN(graph1)
        qctn2 = QCTN(graph2)
        
        result1 = self.engine.contract_core_only(qctn1)
        result2 = self.engine.contract_core_only(qctn2)
        
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        # Results should have different shapes if QCTN structures differ
        self.assertTrue(hasattr(result1, 'shape'))
        self.assertTrue(hasattr(result2, 'shape'))

    def test_sequential_contractions(self):
        """Test sequential contractions with same QCTN."""
        graph_string = "-2-A-2-\n-2-A-2-"
        qctn = QCTN(graph_string)
        
        # Multiple contractions
        result1 = self.engine.contract_core_only(qctn)
        result2 = self.engine.contract_core_only(qctn)
        result3 = self.engine.contract_core_only(qctn)
        
        # All results should be equal (deterministic)
        np.testing.assert_array_equal(np.array(result1), np.array(result2))
        np.testing.assert_array_equal(np.array(result2), np.array(result3))


if __name__ == '__main__':
    unittest.main()
