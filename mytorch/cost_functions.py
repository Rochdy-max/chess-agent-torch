import numpy as np

class CostFunctionsFactory:
    """
    Provides cost functions helping to determine the accuracy of a neural network
    """

    @staticmethod
    def deliver(fname: str):
        """
        Return a function from its identifier
        """
        match fname:
            case "cross_entropy":
                return CostFunctionsFactory._cross_entropy
            case _:
                raise ValueError(f"Unknown cost function: {fname}")

    @staticmethod
    def _cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Cross Entropy Loss function, typically serving
        multi-class and multi-label classifications.
            
        J = -SUM(T * S)
            
        T := The expected target from y
        S := The corresponding prediction (or approximation) from y_hat
        """
        sum_val = 0
        for t, s in zip(y, y_hat):
            sum_val += np.dot(t, np.log(s) / np.log(10))
        return -sum_val
