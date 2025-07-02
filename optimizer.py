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
                       epsilon=1e-8 # Small constant to prevent division by zero
                  ):

        self.method = method
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0

    def optimize(self, qctn, target_qctn):
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
            qctn.cores_weights[c] -= self.learning_rate * grads[idx]


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
            qctn.m = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}
            qctn.v = {c: jnp.zeros_like(w) for c, w in qctn.cores_weights.items()}

        for idx, c in enumerate(qctn.cores):
            # Update biased first moment estimate
            qctn.m[c] = self.beta1 * qctn.m[c] + (1 - self.beta1) * grads[idx]
            
            # Update biased second moment estimate
            qctn.v[c] = self.beta2 * qctn.v[c] + (1 - self.beta2) * (grads[idx] ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = qctn.m[c] / (1 - self.beta1 ** (self.iter + 1))
            v_hat = qctn.v[c] / (1 - self.beta2 ** (self.iter + 1))
            
            # Update parameters
            qctn.cores_weights[c] -= self.learning_rate * m_hat / (jax.numpy.sqrt(v_hat) + self.epsilon)
        
