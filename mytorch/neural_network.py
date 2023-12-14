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
        """
            Applies forward propagation on an input matrix 'X' with lines
            for the activation values of this network input layer's neurons.
            
            Each column of this matrix should correspond to an example.
            
            The returned matrix 'y_hat' contains the has the same architecture.
            Each line is bound to a neuron of the last layer and the
            columns are bound to examples.
        """
        invalid_input_size_error_msg = "The input matrix 'X' should have the same number of lines as the number of neurons of the network's first layer"        

        if X.shape[0] != self.layers[0].shape[1]:
            error_details = f"Number of lines in X ({X.shape[0]}) != Number of neurons in input layer ({self.layers[0].shape[1]})"
            raise ValueError(invalid_input_size_error_msg, error_details)

        A = X
        for layer in self.layers:
            A = layer.process(A)
        # The last layer activation matrix (A) is also y_hat
        return A

    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
            Cost function determining the accuracy of a neural network.
            It uses the Cross Entropy Loss function.

            J = -SUM(T * log(S))
            
            S := A prediction (or approximation) from y_hat
            T := The corresponding expected target from y
        """
        distinct_matrices_shape_error_msg = "y and y_hat should have the same shape"
        sum_val = 0

        if y.shape != y_hat.shape:
            error_details = f"{y.shape} != {y_hat.shape}"
            raise ValueError(distinct_matrices_shape_error_msg, error_details)

        # For each example (column which is now row in transposed matrix),
        # add result of the computation S * log(T) to the total_sum
        for s, t in zip(y.T, y_hat.T):
            sum_val += np.dot(s, np.log(t))

        return -sum_val
            
    
    def back_prop(self, y: np.ndarray) -> None:
        pass
