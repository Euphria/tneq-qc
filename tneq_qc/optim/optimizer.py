import random
import torch
import jax
import jax.numpy as jnp

class Optimizer:
    """
    Optimizer class for JAX-based optimization tasks.
    
    This class provides methods to optimize functions using JAX's optimization capabilities.
    """

    def __init__(self, method='adam', 
                       learning_rate=0.01, 
                       max_iter=1000, 
                       tol=1e-6, # Tolerance for convergence
                       beta1=0.9, # Adam's first moment estimate decay rate
                       beta2=0.999, # Adam's second moment estimate decay rate
                       epsilon=1e-8, # Small constant to prevent division by zero
                       executor=None,
                       # SGDG parameters
                       momentum=0.0, # Momentum factor for SGDG
                       stiefel=True, # Whether to use Stiefel manifold optimization
                  ):

        self.method = method
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0
        self.momentum = momentum
        self.stiefel = stiefel

        self.executor = executor

    def optimize(self, qctn, data_list, **kwargs):
        """
        Optimize a function.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            kwargs: Additional arguments for different optimization modes.

        Returns:
            None: The function modifies the qctn in place.
        """
        while self.iter < self.max_iter:
            # TODO: impl general function named contract_for_gradient
            data_index = self.iter % len(data_list)
            # loss, grads = self.executor.contract_with_self_for_gradient(qctn, **data_list[data_index], **kwargs)
            loss, grads = self.executor.contract_with_std_graph_for_gradient(qctn, **data_list[data_index], **kwargs)

            # Convert loss to scalar for comparison and printing
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            
            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break
            
            print(f"Iteration {self.iter}: loss = {loss_value}")

            # Update parameters using the optimizer step
            with torch.no_grad():
                cache_lr = self.learning_rate

                if self.iter < 1000:
                    max_grad = 0.0
                    for i in range(len(grads)):
                        grad = grads[i].abs().max()
                        if grad > max_grad:
                            max_grad = grad
                
                    if max_grad < 1e-2:
                        self.learning_rate = self.learning_rate * 1e-1 / (max_grad + 1e-30)
                        
                qctn.params = self.step(qctn, grads)

                self.learning_rate = cache_lr

            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_debug(self, qctn, data_list, **kwargs):
        """
        Optimize a function.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            kwargs: Additional arguments for different optimization modes.

        Returns:
            None: The function modifies the qctn in place.
        """
        debug = True
        while self.iter < self.max_iter:
            # if debug and self.iter % 10 == 0:
            #     for i in range(len(qctn.cores_weights)):
            #         print(f"\nCore {i} shape: {qctn.cores_weights[qctn.cores[i]].shape}")
            #         print(f"\nCore {i} weights: {qctn.cores_weights[qctn.cores[i]].detach().cpu().numpy()}")

            # TODO: impl general function named contract_for_gradient
            data_index = self.iter % len(data_list)
            loss, grads = self.executor.contract_with_self_for_gradient(qctn, **data_list[data_index], **kwargs)
            
            # if debug and self.iter % 10 == 0:
            #     # for i in range(len(qctn.cores_weights)):
            #     for i in range(1):
            #         print(f"\ngrad {i} shape: {grads[i].shape}")
            #         print(f"\ngrad {i} weight: {grads[i].detach().cpu().numpy()}")

            # Convert loss to scalar for comparison and printing
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            
            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break
            
            print(f"Iteration {self.iter}: loss = {loss_value}")

            # Update parameters using the optimizer step
            with torch.no_grad():
                cache_lr = self.learning_rate
                if self.iter < 1000:
                    max_grad = 0.0
                    for i in range(len(grads)):
                        grad = grads[i].abs().max()
                        if grad > max_grad:
                            max_grad = grad
                
                    if max_grad < 1e-5:
                        # self.learning_rate = self.learning_rate * 1e-2 / (max_grad + 1e-30)
                        self.learning_rate = self.learning_rate / (max_grad + 1e-30) * 1e-9
                    
                    # if debug and self.iter % 10 == 0:
                    print('max_grad:', max_grad, self.learning_rate)

                # if self.iter < 1000:
                #     max_grad = 0.0
                #     for i in range(len(grads)):
                #         grad = grads[i].abs().max()
                #         if grad > max_grad:
                #             max_grad = grad
                #     if max_grad < 1e-3:
                #         for i in range(len(grads)):
                #             grads[i] = grads[i] / (max_grad + 1e-30) * 0.1
                            

                #     if debug and self.iter % 10 == 0:
                #         print('max_grad:', max_grad)
                #         # for i in range(len(qctn.cores_weights)):
                #         # for i in range(1):
                #         #     print(f"\nnorm grad {i} shape: {grads[i].shape}, {grads[i].dtype}")
                #         #     print(f"\nnorm grad {i} weight: {grads[i].detach().cpu().numpy()}")

                qctn.params = self.step(qctn, grads)

                self.learning_rate = cache_lr
            self.iter += 1

            # if debug and self.iter % 10 == 1:
            #     for i in range(len(qctn.cores_weights)):
            #         print(f"\nres {i} shape: {qctn.cores_weights[qctn.cores[i]].shape}")
            #         print(f"\nres {i} weights: {qctn.cores_weights[qctn.cores[i]].detach().cpu().numpy()}")

        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_with_target(self, qctn, target_qctn):
        """
        Optimize a function using JAX.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            target_qctn (QCTN): The target quantum circuit tensor network for optimization.

        Returns:
            None: The function modifies the qctn in place.
        """

        while self.iter < self.max_iter:
            loss, grads = qctn.contract_with_QCTN_for_gradient(target_qctn)
            if loss < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss}.")
                break

            # Update parameters using the optimizer step
            qctn.params = self.step(qctn, grads)
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss}.")

    def optimize_self_with_inputs(self, qctn, inputs_list):
        """
        Optimize a function using JAX with self-contraction and given inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            inputs_list (list): List of input arrays for the contraction.

        Returns:
            None: The function modifies the qctn in place.
        """

        input_index_list = list(range(len(inputs_list)))
        # shuffle input_index_list
        train_index_list = random.sample(input_index_list, len(input_index_list))
        print(f"train_index_list : {train_index_list}")

        while self.iter < self.max_iter:
            inputs = inputs_list[train_index_list[self.iter % len(inputs_list)]]

            loss, grads = qctn.contract_with_self_for_gradient(inputs)
            if loss < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss}.")
                break

            # Update parameters using the optimizer step
            qctn.params = self.step(qctn, grads)
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss}.")


    def step(self, qctn, grads):
        """
        Perform a single optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """

        if self.method == 'adam':
            self.adam_step(qctn, grads)
        elif self.method == 'sgd':
            self.sgd_step(qctn, grads)
        elif self.method == 'sgdg':
            self.sgdg_step(qctn, grads)
        elif self.method == 'momentum':
            self.momentum_step(qctn, grads)
        elif self.method == 'nesterov':
            self.nesterov_step(qctn, grads)
        elif self.method == 'rmsprop':
            self.rmsprop_step(qctn, grads)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

    #### GPT generated methods for different optimization algorithms ####
    def rmsprop_step(self, qctn, grads):
        """
        Perform a single RMSProp optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        # Initialize cache if not already done
        if not hasattr(qctn, 'cache'):
            qctn.cache = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}

        for idx, c in enumerate(qctn.cores):
            # Update cache
            qctn.cache[c] = 0.9 * qctn.cache[c] + 0.1 * (grads[idx] ** 2)
            # Update parameters
            qctn.cores_weights[c] -= self.learning_rate * grads[idx] / (jnp.sqrt(qctn.cache[c]) + self.epsilon)

    def nesterov_step(self, qctn, grads):
        """
        Perform a single Nesterov accelerated gradient descent step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        # Initialize momentum if not already done
        if not hasattr(qctn, 'momentum'):
            qctn.momentum = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}

        for idx, c in enumerate(qctn.cores):
            # Update momentum
            qctn.momentum[c] = 0.9 * qctn.momentum[c] + self.learning_rate * grads[idx]
            # Update parameters with Nesterov acceleration
            qctn.cores_weights[c] -= qctn.momentum[c] + self.learning_rate * grads[idx]

    def momentum_step(self, qctn, grads):
        """
        Perform a single Momentum optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        # Initialize momentum if not already done
        if not hasattr(qctn, 'momentum'):
            qctn.momentum = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}

        for idx, c in enumerate(qctn.cores):
            # Update momentum
            qctn.momentum[c] = 0.9 * qctn.momentum[c] + self.learning_rate * grads[idx]
            # Update parameters
            qctn.cores_weights[c] -= qctn.momentum[c]

    def sgd_step(self, qctn, grads):
        """
        Perform a single Stochastic Gradient Descent (SGD) optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        for idx, c in enumerate(qctn.cores):
            qctn.cores_weights[c] = qctn.cores_weights[c] - self.learning_rate * grads[idx]


    def adam_step(self, qctn, grads):
        """
        Perform a single Adam optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        # Initialize moment estimates
        if not hasattr(qctn, 'm'):
            if self.executor.backend.get_backend_name() == 'jax':
                qctn.m = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}
                qctn.v = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}
            elif self.executor.backend.get_backend_name() == 'pytorch':
                device = self.executor.backend.backend_info.device

                qctn.m = {c: torch.zeros_like(w, device=device) for c, w in qctn.cores_weights.items()}
                qctn.v = {c: torch.zeros_like(w, device=device) for c, w in qctn.cores_weights.items()}



        for idx, c in enumerate(qctn.cores):

            # Update biased first moment estimate
            qctn.m[c] = self.beta1 * qctn.m[c] + (1 - self.beta1) * grads[idx]
            
            # Update biased second moment estimate
            qctn.v[c] = self.beta2 * qctn.v[c] + (1 - self.beta2) * (grads[idx] ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = qctn.m[c] / (1 - self.beta1 ** (self.iter + 1))
            v_hat = qctn.v[c] / (1 - self.beta2 ** (self.iter + 1))
            
            sqrt_v_hat = jnp.sqrt(v_hat) if self.executor.backend.get_backend_name() == 'jax' else torch.sqrt(v_hat)
            
            # Update parameters (create new tensor instead of in-place operation)
            update = self.learning_rate * m_hat / (sqrt_v_hat + self.epsilon)
            qctn.cores_weights[c] = qctn.cores_weights[c] - update

    def sgdg_step(self, qctn, grads):
        """
        Perform a single SGDG (SGD on Stiefel manifold) optimization step.
        
        This method updates parameters while maintaining orthogonality constraints
        using Cayley transform on the Stiefel manifold.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        device = self.executor.backend.backend_info.device
        
        # Initialize momentum buffers if not already done
        if not hasattr(qctn, 'momentum_buffer'):
            qctn.momentum_buffer = {}
        
        epsilon = 1e-8
        
        for idx, c in enumerate(qctn.cores):
            grad = grads[idx]
            param = qctn.cores_weights[c]
            
            # Check if parameter satisfies Stiefel constraint (rows <= cols after reshape)
            param_shape = param.shape
            
            # For tensors with more than 2 dimensions, reshape to matrix
            if len(param_shape) > 2:
                # Reshape similar to SGDG: flatten first dimensions
                flat_dim = torch.prod(torch.tensor(param_shape[:len(param_shape)//2]))
                param_2d = param.reshape(flat_dim, -1)
                grad_2d = grad.reshape(flat_dim, -1)
            else:
                param_2d = param
                grad_2d = grad
            
            # Normalize to get orthogonal matrix
            unity, unity_norm = self._unit(param_2d)
            
            # Check if we should use Stiefel optimization
            if self.stiefel and unity.shape[0] <= unity.shape[1]:
                # Randomly apply QR retraction for numerical stability (1% chance)
                if random.randint(1, 101) == 1:
                    unity = self._qr_retraction(unity)
                
                # Initialize momentum buffer for this core
                if c not in qctn.momentum_buffer:
                    qctn.momentum_buffer[c] = torch.zeros(grad_2d.T.shape, device=device)
                
                V = qctn.momentum_buffer[c]
                
                # Update momentum: V = momentum * V - g^T
                V = self.momentum * V - grad_2d.T
                
                # Compute the skew-symmetric matrix W
                MX = torch.mm(V, unity)
                XMX = torch.mm(unity, MX)
                XXMX = torch.mm(unity.T, XMX)
                
                W_hat = MX - 0.5 * XXMX
                W = W_hat - W_hat.T  # Make it skew-symmetric
                
                # Compute adaptive step size
                W_norm = self._matrix_norm_one(W)
                t = 0.5 * 2 / (W_norm + epsilon)
                alpha = min(t, self.learning_rate)
                
                # Apply Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
                p_new = self._compute_cayley_transform(alpha, W, unity.T).T
                
                # Reshape back to original shape
                if len(param_shape) > 2:
                    p_new = p_new.reshape(param_shape)
                
                # Update parameter
                qctn.cores_weights[c] = p_new
                
                # Update momentum buffer
                V_new = torch.mm(W, unity.T)
                qctn.momentum_buffer[c] = V_new
                
            else:
                # Standard SGD update for non-Stiefel parameters
                qctn.cores_weights[c] = param - self.learning_rate * grad
    
    # Helper functions for SGDG
    def _unit(self, v, dim=1, eps=1e-8):
        """Normalize a matrix to have unit norm."""
        vnorm = torch.norm(v, p=2, dim=dim, keepdim=True)
        return v / (vnorm + eps), vnorm
    
    def _qr_retraction(self, tan_vec):
        """QR retraction to project back onto Stiefel manifold."""
        tan_vec_T = tan_vec.T
        q, r = torch.linalg.qr(tan_vec_T, mode='reduced')
        d = torch.diag(r)
        ph = torch.sign(d)
        q = q * ph.unsqueeze(0)
        return q.T
    
    def _matrix_norm_one(self, W):
        """Compute matrix 1-norm (maximum absolute column sum)."""
        return torch.abs(W).sum(dim=0).max()
    
    def _compute_cayley_transform(self, alpha, W, X):
        """
        Compute Cayley transform: Y(alpha) = (I - alpha/2 * W)^{-1} (I + alpha/2 * W) X
        
        Args:
            alpha: Step size
            W: Skew-symmetric matrix
            X: Current point on manifold
            
        Returns:
            Updated point Y(alpha)
        """
        I = torch.eye(W.shape[0], device=W.device)
        left_matrix = I - (alpha / 2) * W
        right_matrix = I + (alpha / 2) * W
        left_inv = torch.inverse(left_matrix)
        Y_alpha = left_inv @ right_matrix @ X
        
        return Y_alpha
        
