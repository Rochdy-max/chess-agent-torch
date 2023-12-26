import numpy as np
from utils import array_map

class ActivationFunctionsFactory:
    """
        Provide activation functions which applied to the weighted sum matrix ('z') 
        should determine the activations for a layer's neurons and return a matrix
        containing these activations.
    """

    @staticmethod
    def deliver(fname: str):
        """
        Return a function from its identifier
        """
        match fname:
            case "relu":
                return ActivationFunctionsFactory._relu
            case "sigmoid":
                return ActivationFunctionsFactory._sigmoid
            case _:
                raise ValueError(f"Unknown activation function: {fname}")

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        """
            ReLU function
            
            f(x) = max(0, x)
        """
        return array_map(lambda x: max(0, x), z)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
            Sigmoid function
            
            f(x) = 1 / (1 + exp(-z))
        """
        return 1 / (1 + np.exp(-z))
