from typing import Any

class TNTensor:
    """
    Tensor Network Tensor class that wraps a backend tensor and a scale factor.
    
    This class allows for high-precision scaling of tensors to avoid underflow/overflow
    issues during tensor network contractions.
    """
    
    def __init__(self, tensor: Any, scale: Any = 1.0):
        """
        Initialize TNTensor.
        
        Args:
            tensor: The backend tensor (PyTorch tensor, JAX array, or NumPy array).
            scale: The scaling factor (float or tensor-like).
        """
        self._tensor = tensor
        
        # Initialize scale as a tensor compatible with self._tensor if possible
        if hasattr(self._tensor, 'new_tensor'): # Likely PyTorch
             if not hasattr(scale, 'shape') or len(scale.shape) == 0:
                 # Check if scale is already a tensor
                 if hasattr(scale, 'to') and hasattr(scale, 'device'):
                     self.scale = scale.to(self._tensor.device)
                 else:
                     self.scale = self._tensor.new_tensor(scale)
             else:
                 self.scale = self._tensor.new_tensor(scale)
        else:
             self.scale = scale

    @property
    def tensor(self) -> Any:
        """Get the underlying backend tensor."""
        return self._tensor

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the underlying tensor."""
        return self._tensor.ndim

    @property
    def shape(self) -> tuple:
        """Get the shape of the underlying tensor."""
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        """Get the dtype of the underlying tensor."""
        return self._tensor.dtype

    def auto_scale(self):
        """
        Automatically scale the tensor so that its absolute max value is 1.
        Updates self.scale accordingly.
        """
        # Calculate the maximum absolute value
        # We use abs() and max() which should work for PyTorch, JAX, and NumPy
        max_val = self._tensor.abs().max()
        
        # Convert to float for the scale calculation
        # .item() works for PyTorch, JAX, and NumPy scalars
        if hasattr(max_val, 'item'):
            max_val_float = max_val.item()
        else:
            max_val_float = float(max_val)
            
        if max_val_float == 0:
            return # Tensor is all zeros, no scaling needed

        # Update scale
        self.scale *= max_val_float
        
        # Normalize tensor
        self._tensor = self._tensor / max_val_float

    def scale_to(self, new_scale: float):
        """
        Scale the tensor to a new scale value.
        The actual represented value (tensor * scale) remains unchanged.
        
        Args:
            new_scale (float): The new scale value.
        """
        new_scale = float(new_scale)
        if new_scale == 0:
             raise ValueError("Cannot scale to 0.")
             
        factor = self.scale / new_scale
        self._tensor = self._tensor * factor
        self.scale = new_scale

    def scale_with(self, factor: float):
        """
        Multiply the scale by a factor, and divide the tensor by the same factor.
        The actual represented value remains unchanged.
        
        Args:
            factor (float): The factor to scale with.
        """
        factor = float(factor)
        if factor == 0:
            raise ValueError("Cannot scale with factor 0.")
            
        self.scale *= factor
        self._tensor = self._tensor / factor

    def scale_to(self, new_scale: float):
        """
        Scale the tensor to a new scale value.
        The actual represented value (tensor * scale) remains unchanged.
        
        Args:
            new_scale (float): The new scale value.
        """
        new_scale = float(new_scale)
        if new_scale == 0:
             raise ValueError("Cannot scale to 0.")
             
        factor = self.scale / new_scale
        self._tensor = self._tensor * factor
        self.scale = new_scale

    def scale_with(self, factor: float):
        """
        Multiply the scale by a factor, and divide the tensor by the same factor.
        The actual represented value remains unchanged.
        
        Args:
            factor (float): The factor to scale with.
        """
        factor = float(factor)
        if factor == 0:
            raise ValueError("Cannot scale with factor 0.")
            
        self.scale *= factor
        self._tensor = self._tensor / factor

    def __repr__(self):
        shape = getattr(self._tensor, 'shape', 'unknown')
        return f"TNTensor(shape={shape}, scale={self.scale})"
