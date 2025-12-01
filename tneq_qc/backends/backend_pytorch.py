"""
PyTorch backend implementation.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import random

from .backend_interface import ComputeBackend, BackendInfo


class BackendPyTorch(ComputeBackend):
    """PyTorch computational backend."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch backend.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects 'cuda' if available, otherwise 'cpu'.
        """
        super().__init__()
        try:
            import torch
            self.torch = torch
            
            # Auto-detect device if not specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create BackendInfo
            self.backend_info = BackendInfo('pytorch', device=device)
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it with: pip install torch")

    def execute_expression(self, expression, *tensors):
        """
        Execute contraction expression using PyTorch.
        
        Note: opt_einsum's ContractExpression works with PyTorch tensors.
        """
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Compute value and gradient using PyTorch autograd.
        
        Returns a function that computes loss and gradients for specified arguments.
        """
        def value_and_grad_fn(*args):
            # Convert argnums to list if it's a range
            if isinstance(argnums, range):
                arg_indices = list(argnums)
            elif isinstance(argnums, (list, tuple)):
                arg_indices = argnums
            else:
                arg_indices = [argnums]

            # Prepare tensors and enable gradients for specified arguments
            tensor_args = []
            for i, arg in enumerate(args):
                if not isinstance(arg, self.torch.Tensor):
                    arg = self.torch.from_numpy(arg).float()
                
                if i in arg_indices:
                    # Enable gradient tracking for parameters to optimize
                    arg.requires_grad_(True)
                else:
                    # Disable gradient tracking for other tensors
                    arg.requires_grad_(False)
                
                tensor_args.append(arg)

            # Compute loss
            loss = loss_fn(*tensor_args)

            # Compute gradients using torch.autograd.grad
            # This works correctly with both leaf and non-leaf tensors
            grad_tensors = [tensor_args[i] for i in arg_indices]
            gradients = self.torch.autograd.grad(
                outputs=loss,
                inputs=grad_tensors,
                create_graph=False,  # Don't build computation graph for gradients
                retain_graph=False   # Release computation graph after computing gradients
            )

            # Return loss as tensor value (not .item()) to preserve type
            # Detach it from computation graph to free memory
            return loss.detach(), gradients

        return value_and_grad_fn

    def jit_compile(self, func):
        """
        JIT compile function using PyTorch.
        
        Note: PyTorch's torch.jit.script has limitations.
        For now, we return the function as-is and rely on PyTorch's built-in optimizations.
        """
        # Could use torch.jit.trace or torch.compile in PyTorch 2.0+
        return func

    def convert_to_tensor(self, array):
        """Convert to PyTorch tensor."""
        if isinstance(array, self.torch.Tensor):
            # Move to correct device if needed
            if str(array.device) != self.backend_info.device:
                return array.to(self.backend_info.device)
            return array
        
        # Convert array to tensor based on backend_info
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        tensor = self.torch.from_numpy(array).float()
        return tensor.to(self.backend_info.device)

    def get_backend_name(self) -> str:
        return "pytorch"

    def optimizer_update(self, params: List[Any], grads: List[Any], state: Dict[str, Any], 
                        method: str, hyperparams: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Perform a single optimization step using PyTorch.
        """
        with self.torch.no_grad():
            if method == 'adam':
                return self._adam_step(params, grads, state, hyperparams)
            elif method == 'sgd':
                return self._sgd_step(params, grads, state, hyperparams)
            elif method == 'sgdg':
                return self._sgdg_step(params, grads, state, hyperparams)
            elif method == 'momentum':
                return self._momentum_step(params, grads, state, hyperparams)
            elif method == 'nesterov':
                return self._nesterov_step(params, grads, state, hyperparams)
            elif method == 'rmsprop':
                return self._rmsprop_step(params, grads, state, hyperparams)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

    def _adam_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        beta1 = hp.get('beta1', 0.9)
        beta2 = hp.get('beta2', 0.999)
        epsilon = hp.get('epsilon', 1e-8)
        iteration = hp.get('iter', 0)

        if 'm' not in state:
            state['m'] = [self.torch.zeros_like(p) for p in params]
            state['v'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            state['m'][i] = beta1 * state['m'][i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            state['v'][i] = beta2 * state['v'][i] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = state['m'][i] / (1 - beta1 ** (iteration + 1))
            v_hat = state['v'][i] / (1 - beta2 ** (iteration + 1))
            
            sqrt_v_hat = self.torch.sqrt(v_hat)
            
            # Update parameters
            update = lr * m_hat / (sqrt_v_hat + epsilon)
            new_params.append(param - update)
            
        return new_params, state

    def _sgd_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        new_params = []
        for param, grad in zip(params, grads):
            new_params.append(param - lr * grad)
        return new_params, state

    def _momentum_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        momentum = hp.get('momentum', 0.0) # Default 0.0 if not provided, though usually 0.9

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['momentum_buffer'][i] = 0.9 * state['momentum_buffer'][i] + lr * grad
            new_params.append(param - state['momentum_buffer'][i])
            
        return new_params, state

    def _nesterov_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['momentum_buffer'][i] = 0.9 * state['momentum_buffer'][i] + lr * grad
            new_params.append(param - (state['momentum_buffer'][i] + lr * grad))
            
        return new_params, state

    def _rmsprop_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        epsilon = hp.get('epsilon', 1e-8)

        if 'square_avg' not in state:
            state['square_avg'] = [self.torch.zeros_like(p) for p in params]

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            state['square_avg'][i] = 0.9 * state['square_avg'][i] + 0.1 * (grad ** 2)
            new_params.append(param - lr * grad / (self.torch.sqrt(state['square_avg'][i]) + epsilon))
            
        return new_params, state

    def _sgdg_step(self, params, grads, state, hp):
        lr = hp.get('learning_rate', 0.01)
        momentum = hp.get('momentum', 0.0)
        stiefel = hp.get('stiefel', True)
        epsilon = 1e-8

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = [None] * len(params)

        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Check if parameter satisfies Stiefel constraint (rows <= cols after reshape)
            param_shape = param.shape
            
            # For tensors with more than 2 dimensions, reshape to matrix
            if len(param_shape) > 2:
                # Reshape similar to SGDG: flatten first dimensions
                flat_dim = int(np.prod(param_shape[:len(param_shape)//2]))
                param_2d = param.reshape(flat_dim, -1)
                grad_2d = grad.reshape(flat_dim, -1)
            else:
                param_2d = param
                grad_2d = grad
            
            # Normalize to get orthogonal matrix
            unity, unity_norm = self._unit(param_2d)
            
            # Check if we should use Stiefel optimization
            if stiefel and unity.shape[0] <= unity.shape[1]:
                # Randomly apply QR retraction for numerical stability (1% chance)
                if random.randint(1, 101) == 1:
                    unity = self._qr_retraction(unity)
                
                # Initialize momentum buffer for this core
                if state['momentum_buffer'][i] is None:
                    state['momentum_buffer'][i] = self.torch.zeros(grad_2d.T.shape, device=param.device)
                
                V = state['momentum_buffer'][i]
                
                # Update momentum: V = momentum * V - g^T
                V = momentum * V - grad_2d.T
                
                # Compute the skew-symmetric matrix W
                MX = self.torch.mm(V, unity)
                XMX = self.torch.mm(unity, MX)
                XXMX = self.torch.mm(unity.T, XMX)
                
                W_hat = MX - 0.5 * XXMX
                W = W_hat - W_hat.T  # Make it skew-symmetric
                
                # Compute adaptive step size
                W_norm = self._matrix_norm_one(W)
                t = 0.5 * 2 / (W_norm + epsilon)
                alpha = min(t, lr)
                
                # Apply Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
                p_new = self._compute_cayley_transform(alpha, W, unity.T).T
                
                # Reshape back to original shape
                if len(param_shape) > 2:
                    p_new = p_new.reshape(param_shape)
                
                new_params.append(p_new)
                
                # Update momentum buffer
                V_new = self.torch.mm(W, unity.T)
                state['momentum_buffer'][i] = V_new
                
            else:
                # Standard SGD update for non-Stiefel parameters
                new_params.append(param - lr * grad)
        
        return new_params, state

    # Helper functions for SGDG
    def _unit(self, v, dim=1, eps=1e-8):
        """Normalize a matrix to have unit norm."""
        vnorm = self.torch.norm(v, p=2, dim=dim, keepdim=True)
        return v / (vnorm + eps), vnorm
    
    def _qr_retraction(self, tan_vec):
        """QR retraction to project back onto Stiefel manifold."""
        tan_vec_T = tan_vec.T
        q, r = self.torch.linalg.qr(tan_vec_T, mode='reduced')
        d = self.torch.diag(r)
        ph = self.torch.sign(d)
        q = q * ph.unsqueeze(0)
        return q.T
    
    def _matrix_norm_one(self, W):
        """Compute matrix 1-norm (maximum absolute column sum)."""
        return self.torch.abs(W).sum(dim=0).max()
    
    def _compute_cayley_transform(self, alpha, W, X):
        """
        Compute Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
        """
        I = self.torch.eye(W.shape[0], device=W.device)
        left_matrix = I - (alpha / 2) * W
        right_matrix = I + (alpha / 2) * W
        left_inv = self.torch.inverse(left_matrix)
        Y_alpha = left_inv @ right_matrix @ X
        
        return Y_alpha
