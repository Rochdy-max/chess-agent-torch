import numpy as np
from utils import array_map

class ActivationDerivatives:
    """
        Provide derivatives of the activation functions for the implementation of back propagation
    """

    def relu_prime(z: np.ndarray):
        """
            Derivative of the ReLU activation function
            
            f'(x) = 1 if the value is positive and 0 either
        """
        return array_map(lambda x: float(x > 0), z)
