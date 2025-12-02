"""
Backend JAX implementation.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .backend_interface import ComputeBackend, BackendInfo


class BackendJAX(ComputeBackend):
    """JAX computational backend."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize backend JAX.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'gpu', etc.).
                If None, automatically detects available devices.
        """
        super().__init__()
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            
            # Auto-detect device if not specified
            if device is None:
                device = 'gpu' if jax.devices('gpu') else 'cpu'
            
            # Create BackendInfo
            self.backend_info = BackendInfo('jax', device=device)
            
            # Initialize PRNG key
            self.key = self.jax.random.PRNGKey(0)
        except ImportError:
            raise ImportError("JAX is not installed. Please install it with: pip install jax jaxlib")

    def execute_expression(self, expression, *tensors):
        """Execute contraction expression using JAX."""
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """Compute value and gradient using JAX autodiff."""
        return self.jax.value_and_grad(loss_fn, argnums=argnums)

    def jit_compile(self, func):
        """JIT compile function using JAX."""
        return self.jax.jit(func)

    def convert_to_tensor(self, array):
        """Convert to JAX array."""
        if isinstance(array, self.jnp.ndarray):
            return array
        
        # Convert based on backend_info
        tensor = self.jnp.array(array)
        
        # Device placement if GPU
        if self.backend_info.device and 'gpu' in self.backend_info.device.lower():
            devices = self.jax.devices('gpu')
            if devices:
                tensor = self.jax.device_put(tensor, devices[0])
        
        return tensor

    def get_backend_name(self) -> str:
        return "jax"

    def optimizer_update(self, params: List[Any], grads: List[Any], state: Dict[str, Any], 
                        method: str, hyperparams: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Perform a single optimization step using JAX.
        """
        if method == 'adam':
            return self._adam_step(params, grads, state, hyperparams)
        elif method == 'sgd':
            return self._sgd_step(params, grads, state, hyperparams)
        else:
            raise NotImplementedError(f"Optimization method {method} not implemented for JAX backend yet.")

    def _adam_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        beta1 = hp.get('beta1', 0.9)
        beta2 = hp.get('beta2', 0.999)
        epsilon = hp.get('epsilon', 1e-8)
        iteration = hp.get('iter', 0)

        if 'm' not in state:
            state['m'] = [self.jnp.zeros_like(p) for p in params]
            state['v'] = [self.jnp.zeros_like(p) for p in params]

        new_params = []
        new_m = []
        new_v = []
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            m = state['m'][i]
            v = state['v'][i]
            
            # Update biased first moment estimate
            m_new = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment estimate
            v_new = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = m_new / (1 - beta1 ** (iteration + 1))
            v_hat = v_new / (1 - beta2 ** (iteration + 1))
            
            sqrt_v_hat = self.jnp.sqrt(v_hat)
            
            # Update parameters
            update = lr * m_hat / (sqrt_v_hat + epsilon)
            new_params.append(param - update)
            
            new_m.append(m_new)
            new_v.append(v_new)
            
        new_state = state.copy()
        new_state['m'] = new_m
        new_state['v'] = new_v
            
        return new_params, new_state

    def _sgd_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        new_params = []
        for param, grad in zip(params, grads):
            new_params.append(param - lr * grad)
        return new_params, state

    def init_random_core(self, shape):
        flat_dim = int(np.prod(shape[:len(shape)//2]))
        
        # Split key
        self.key, subkey = self.jax.random.split(self.key)
        
        random_matrix = self.jax.random.normal(subkey, (flat_dim, flat_dim))
        Q, R = self.jnp.linalg.qr(random_matrix)
        d = self.jnp.diag(R)
        sign_correction = self.jnp.sign(d)
        Q = Q * sign_correction[None, :] # unsqueeze(0) equivalent
        return Q.reshape(shape)

    def get_tensor_type(self):
        return self.jnp.ndarray

    def set_random_seed(self, seed: int):
        self.key = self.jax.random.PRNGKey(seed)
        np.random.seed(seed)

    def tensor_to_numpy(self, tensor):
        return np.asarray(tensor)
