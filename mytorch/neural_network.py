import numpy as np

class NeuralNetwork:
    def __init__(self, layers_count: int, *sizes: int) -> None:
        pass
    
    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def activation(self, z: np.ndarray) -> np.ndarray:
        pass
    
    def compute_cost(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        pass
    
    def back_prop(self, y: np.ndarray) -> None:
        pass
