import numpy as np
from activation_functions import ActivationFunctionsFactory
from utils import array_map

class ActivationDerivativesFactory:
    """
    Provide derivatives of activation functions for the implementation of back propagation
    """

    @staticmethod
    def deliver(fname: str):
        """
        Return a function from its identifier
        """
        match fname:
            case "relu":
                return ActivationDerivativesFactory._relu_prime
            case "sigmoid":
                return ActivationDerivativesFactory._sigmoid_prime
            case _:
                raise ValueError(f"Unknown activation function: {fname}")

    @staticmethod
    def _relu_prime(z: np.ndarray):
        """
        Derivative of the ReLU activation function
            
        f'(x) = 1 if the value is positive and 0 either
        """
        return array_map(lambda x: float(x > 0), z)
    
    @staticmethod
    def _sigmoid_prime(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function
            
        f'(x) = f(x) * (1 - f(x))
        """
        return ActivationFunctionsFactory._sigmoid(z) * (1 - ActivationFunctionsFactory._sigmoid(z))
