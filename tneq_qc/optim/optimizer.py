import random
from typing import Callable, Optional
from ..core.tn_tensor import TNTensor

class Optimizer:
    """
    Optimizer class for optimization tasks.
    
    This class provides methods to optimize functions using the configured backend.
    """

    def __init__(self, method='adam', 
                   learning_rate=0.01, 
                   max_iter=1000, 
                   tol=1e-6, # Tolerance for convergence
                   beta1=0.9, # Adam's first moment estimate decay rate
                   beta2=0.999, # Adam's second moment estimate decay rate
                   epsilon=1e-8, # Small constant to prevent division by zero
                   engine=None,
                   lr_schedule: Optional[list] = None,
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
        self.lr_schedule = lr_schedule

        self.engine = engine
        self.opt_state = {}

    def _apply_lr_schedule(self):
        """Update the current learning rate via lr_schedule if provided.
        
        If lr_schedule is None, the learning_rate remains constant.
        If lr_schedule is provided, it should be a list of (step, lr) tuples,
        sorted by step in ascending order. The learning rate will be set to
        the lr value corresponding to the largest step <= current iteration.
        
        Example:
            lr_schedule = [(0, 1e-2), (200, 1e-3), (800, 1e-4)]
            - step 0-199: lr = 1e-2
            - step 200-799: lr = 1e-3
            - step >= 800: lr = 1e-4
        """
        if self.lr_schedule is None:
            return

        for step, lr in reversed(self.lr_schedule):
            if self.iter >= step:
                self.learning_rate = lr
                return

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
            # loss, grads = self.engine.contract_with_self_for_gradient(qctn, **data_list[data_index], **kwargs)
            # loss, grads = self.engine.contract_with_std_graph_for_gradient(qctn, **data_list[data_index], **kwargs)
            loss, grads = self.engine.contract_with_compiled_strategy_for_gradient(qctn, **data_list[data_index], **kwargs)

            # Convert loss to scalar for comparison and printing
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            
            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break
            
            print(f"Iteration {self.iter}: loss = {loss_value} lr = {self.learning_rate}")

            self.step(qctn, grads)

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
            # TODO: impl general function named contract_for_gradient
            data_index = self.iter % len(data_list)
            loss, grads = self.engine.contract_with_self_for_gradient(qctn, **data_list[data_index], **kwargs)
            
            # Convert loss to scalar for comparison and printing
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break
            
            print(f"Iteration {self.iter}: loss = {loss_value}")

            # Update parameters using the optimizer step
            # Adaptive LR logic - commented out for backend agnosticism
            # if self.iter < 1000:
            #     max_grad = 0.0
            #     for i in range(len(grads)):
            #         grad = grads[i].abs().max()
            #         if grad > max_grad:
            #             max_grad = grad
            # 
            #     if max_grad < 1e-5:
            #         # self.learning_rate = self.learning_rate * 1e-2 / (max_grad + 1e-30)
            #         self.learning_rate = self.learning_rate / (max_grad + 1e-30) * 1e-9
                
            self.step(qctn, grads)

            self.iter += 1

        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_with_target(self, qctn, target_qctn):
        """
        Optimize a function.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            target_qctn (QCTN): The target quantum circuit tensor network for optimization.

        Returns:
            None: The function modifies the qctn in place.
        """

        while self.iter < self.max_iter:
            loss, grads = qctn.contract_with_QCTN_for_gradient(target_qctn)
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            # Update parameters using the optimizer step
            self.step(qctn, grads)
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_self_with_inputs(self, qctn, inputs_list):
        """
        Optimize a function using self-contraction and given inputs.
        
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
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            # Update parameters using the optimizer step
            self.step(qctn, grads)
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")


    def step(self, qctn, grads):
        """
        Perform a single optimization step.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            grads (array-like): The gradients computed from the loss function.

        Returns:
            None: The function modifies the qctn parameters in place.
        """
        # Prepare params list (ensure order matches grads)
        # grads is a list corresponding to qctn.cores
        param_keys = qctn.cores
        params_list = [qctn.cores_weights[k] for k in param_keys]
        
        hyperparams = {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'iter': self.iter,
            'momentum': self.momentum,
            'stiefel': self.stiefel
        }
        
        new_params_list, new_state = self.engine.backend.optimizer_update(
            params_list, grads, self.opt_state, self.method, hyperparams
        )
        
        # Update params in qctn
        for k, p in zip(param_keys, new_params_list):
            qctn.cores_weights[k] = p
            
        self.opt_state = new_state
        
