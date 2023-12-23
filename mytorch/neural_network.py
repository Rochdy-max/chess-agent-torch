import numpy as np
from intersynaptic_space import IntersynapticSpace
from utils import array_map

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

        if A.shape[0] != self.interspaces[0].weights.shape[1]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) isn't equal to number of neurons in neural network's input layer ({self.interspaces[0].weights.shape[1]})")

        for interspace in self.interspaces:
            A = interspace.process(A)

        # The last layer activation matrix (A) is equal to the transposition of y_hat
        return A.T
    
    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
            Cost function determining the accuracy of a neural network.
            It uses the Cross Entropy Loss function, typically serving
            multi-class and multi-label classifications.

            J = -SUM(T * S)
            
            T := The expected target from y
            S := The corresponding prediction (or approximation) from y_hat
        """
        sum_val = 0

        if y.shape != y_hat.shape:
            raise ValueError(f"y and y_hat should have the same shape ({y.shape} != {y_hat.shape})")

        # For each example, add result of the computation S*T to the total_sum
        for t, s in zip(y, y_hat):
            sum_val += np.dot(t, s)

        return -sum_val

    def back_prop(self, y: np.ndarray) -> dict[str, np.ndarray]:
        dA = y.T # derivative of the cost function
        dz = array_map(lambda x: float(x > 0), self.interspaces[-1].A) * dA # derivative of the activation funtion (ReLU) * dA
        grads_W = list()
        grads_b = list()

        for interspace in reversed(self.interspaces):
            dW = np.dot(dz, interspace.X.T) # dW = dz . T(X)
            grads_W.append(dW)
            db = dz # db = dz * 1
            grads_b.append(db)

            dA = np.dot(interspace.weights.T, dz) # prev_dA = T(W) . dz
            dz = array_map(lambda x: float(x > 0), interspace.X) * dA # prev_dZ
            
        return dict([
            ('weights', grads_W),
            ('biases', grads_b)
        ])

    def apply_gradient_descent(self, learning_rate: float, dW: list[np.ndarray], db: list[np.ndarray]) -> None:
        for i in range(len(self.interspaces)):
            # Last In First Out for the gradients
            self.interspaces[i].weights -= learning_rate * dW.pop()
            self.interspaces[i].biases -= learning_rate * db.pop()

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
