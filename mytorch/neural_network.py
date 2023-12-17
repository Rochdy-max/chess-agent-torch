import numpy as np
from intersynaptic_space import IntersynapticSpace

class NeuralNetwork:
    """
        Provides an encapsulation for a neural network
    """

    def __init__(self, *sizes: int) -> None:
        """
            Initializes the characteristic data for a neural network :
            number of layers, weights matrices based on the parameter 'sizes'.
        """
        self.interspaces_count = len(sizes)
        self.interspaces = list()
        for i in range(1, self.interspaces_count):
            self.interspaces.append(IntersynapticSpace(sizes[i], sizes[i - 1]))

    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        """
            Applies forward propagation on an input matrix 'X' with lines
            for the activation values of this network input layer's neurons.
            
            Each line of this matrix should correspond to an example.
            
            The returned matrix 'y_hat' has the same architecture.
            Each column is bound to a neuron of the last layer and the
            columns are bound to examples.
        """
        A = X.T # Using the transposition of X for computation

        if A.shape[0] != self.interspaces[0].shape[1]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) isn't equal to number of neurons in neural network's input layer ({self.interspaces[0].shape[1]})")

        for interspace in self.interspaces:
            A = interspace.process(A)

        # The last layer activation matrix (A) is equal to the transposition of y_hat
        return A.T
    
    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
            Cost function determining the accuracy of a neural network.
            It uses the Cross Entropy Loss function, typically serving
            multi-class and multi-label classifications.

            J = -SUM(T * log(S))
            
            S := A prediction (or approximation) from y_hat
            T := The corresponding expected target from y
        """
        sum_val = 0

        if y.shape != y_hat.shape:
            raise ValueError(f"y and y_hat should have the same shape ({y.shape} != {y_hat.shape})")

        # For each example, add result of the computation S * log(T) to the total_sum
        for s, t in zip(y, y_hat):
            sum_val += np.dot(s, np.log(t))

        return -sum_val

    def back_prop(self, y: np.ndarray) -> np.ndarray:
        pass

    def apply_gradient_descent(self, dW: np.ndarray, db: np.ndarray) -> None:
        pass

def test_compute_cost():
    z = np.zeros((4, 5))
    y = np.random.randn(4, 5)
    y_hat = np.random.randn(4, 5)
    bad_y_hat = np.random.randn(4, 4)

    print(f"{z=}")
    print(f"{y=}")
    print(f"{y_hat=}")

    zeros_images_cost = NeuralNetwork.compute_cost(z, z)
    print(f"{zeros_images_cost=}")

    y_z_images_cost = NeuralNetwork.compute_cost(y, z)
    print(f"{y_z_images_cost=}")

    identical_images_cost = NeuralNetwork.compute_cost(y, y)
    print(f"{identical_images_cost=}")

    different_images_cost = NeuralNetwork.compute_cost(y, y_hat)
    print(f"{different_images_cost=}")

    try:
        cost = NeuralNetwork.compute_cost(y, bad_y_hat)
        print(f"{cost=}")
    except ValueError:
        print("Not same shape")

if __name__ == "__main__":
    test_compute_cost()
