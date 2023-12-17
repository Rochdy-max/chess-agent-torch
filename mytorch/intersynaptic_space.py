import numpy as np
from utils import array_map

class IntersynapticSpace:
    """
        Provides an encapsulation for the data involved between two layers in a neural network
    """

    def __init__(self, size: int, prev_size: int) -> None:
        """
            Initialize the size, weigths matrix and biases vector for this layer.

            Each line from the weights matrix contains the weigths of the
            synapses binding the 'prev_size' neurons of the preceding layer
            to a single neuron of the current layer.

            For the 'size' lines of this matrix (one binding each neuron to
            those of the previous layer) there are 'prev_size' columns. This
            is a "size x prev_size" matrix intialized to random values at this time.
            
            The biases vector also has 'size' lines and each express a value
            to add to the weigthed sum of the previous layer's neurons' activations
            considering the treshold from which this sum should be to activate
            the neuron. At the time, this vector contains only zeros.
        """
        self.size = size
        self.weigths = np.random.randn(size, prev_size)
        self.biases = np.zeros(size)
        self.X = None
        self.A = None

    def activation(self, z: np.ndarray) -> np.ndarray:
        """
            ReLU function applied to the weighted sum matrix ('z') to determine
            activations for the layer's neurons and return a matrix containing
            these activations.
            
            There can be many activations for each neuron if there are multiple
            examples. One column correspond to one example.
        """
        return array_map(lambda x: max(0, x), z)
    
    def process(self, X: np.ndarray) -> np.ndarray:
        """
            Process the input matrix 'X' to determine the activations for the
            layer's neurons and return a matrix containing these activations.

            This operation consist of calculating the weighted sum matrix 'z'
            before passing it to the activation function which will compute
            the neurons' activation as suggested.
            
            'z' and the returned matrix 'A' will have the same number of
            columns as 'X', each corresponding to an example. The number
            of lines will be the same as the number of layer's neurons (which
            is also the number of lines in the weights matrix).
            
            z := w * X + b
            A := activation(z)
        """
        if self.weights.shape[1] != X.shape[0]:
            raise ValueError(f"The number of features ({X.shape[0]}) doesn't match the valid input size ({self.weigths.shape[1]})")

        z = np.dot(self.weigths, X) + self.biases
        self.X = X
        self.A = self.activation(z)
        return self.A
