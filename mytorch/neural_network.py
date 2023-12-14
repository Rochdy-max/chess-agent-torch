import numpy as np
from neurons_layer import NeuronsLayer

class NeuralNetwork:
    """
        Provides an encapsulation for a neural network
    """

    def __init__(self, *sizes: int) -> None:
        """
            Initializes the characteristic data for a neural network :
            number of layers, weights matrices based on the parameter 'sizes'.
        """
        self.layers_count = len(sizes)
        self.layers = list()
        for i in range(1, self.layers_count):
            self.layers.append(NeuronsLayer(sizes[i], sizes[i - 1]))
            
    
    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        pass

    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        pass
            
    def back_prop(self, y: np.ndarray) -> None:
        pass
