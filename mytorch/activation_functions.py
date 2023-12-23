import numpy as np
from utils import array_map

class ActivationFunctions:
    """
        Provide activation functions which applied to the weighted sum matrix ('z') 
        should determine the activations for a layer's neurons and return a matrix
        containing these activations.
    """

    def relu(z: np.ndarray) -> np.ndarray:
        """
            ReLU function
            
            f(x) = max(0, x)
        """
        return array_map(lambda x: max(0, x), z)
