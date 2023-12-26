import numpy as np

class CostDerivativesFactory:
    """
    Provide derivatives of cost functions for the implementation of back propagation
    """

    @staticmethod
    def deliver(fname: str):
        """
        Return a function from its identifier
        """
        match fname:
            case "cross_entropy":
                return CostDerivativesFactory._cross_entropy_prime
            case _:
                raise ValueError(f"Unknown cost function: {fname}")

    @staticmethod
    def _cross_entropy_prime(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Derivative of Cross Entropy Loss function
        
        f'(y, y_hat) = y / (ln(10) * y_hat)
        """
        return -y / (np.log(10) * y_hat)
