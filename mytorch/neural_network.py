import numpy as np
from intersynaptic_space import IntersynapticSpace
from activation_derivatives import ActivationDerivativesFactory
from activation_functions import ActivationFunctionsFactory
from cost_derivatives import CostDerivativesFactory
from cost_functions import CostFunctionsFactory

class NeuralNetwork:
    """
    Provides an encapsulation for a neural network.
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
        self.cache = dict([
            ("A", list()),
            ("z", list())
        ])

    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        """
        Apply forward propagation on an input matrix 'X' with lines
        for the activation values of this network input layer's neurons.

        Each line of this matrix should correspond to an example.

        The returned matrix 'y_hat' has the same architecture.
        Each column is bound to a neuron of the last layer and the
        columns are bound to examples.
        """
        activation_f = ActivationFunctionsFactory.deliver("sigmoid")
        A = np.atleast_2d(X).T # Using the transposition of X for computation

        if A.shape[0] != self.interspaces[0].weights.shape[1]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) isn't equal to number of neurons in neural network's input layer ({self.interspaces[0].weights.shape[1]})")

        self.cache["z"] = []
        self.cache["A"] = [A]
        for interspace in self.interspaces:
            z = interspace.process(A)
            self.cache["z"].append(z)
            A = activation_f(z)
            self.cache["A"].append(A)

        # The last layer activation matrix (A) is equal to the transposition of y_hat
        return A.T
    
    def compute_cost(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Cost function determining the accuracy of a neural network.
        """
        cost_f = CostFunctionsFactory.deliver("cross_entropy")
        y = np.atleast_2d(y)
        y_hat = np.atleast_2d(y_hat)

        if y.shape != y_hat.shape:
            raise ValueError(f"y and y_hat should have the same shape ({y.shape} != {y_hat.shape})")

        return cost_f(y, y_hat)

    def back_prop(self, y: np.ndarray) -> dict[str, list]:
        """
        Return a dictionary containing the gradients to apply to the weights and biases of the network
        """
        activation_prime = ActivationDerivativesFactory.deliver("sigmoid")
        cost_prime = CostDerivativesFactory.deliver("cross_entropy")
        dA = cost_prime(np.atleast_2d(y).T, self.cache["A"][-1]) # derivative of the cost function
        grads_W = list()
        grads_b = list()
        i = -1 # start from the end of the cache

        for interspace in reversed(self.interspaces):
            dz = activation_prime(self.cache["z"][i]) * dA # derivative of the activation funtion * dA
            dW = np.dot(dz, self.cache["A"][i - 1].T) # dW = dz . T(A_prev)
            grads_W.append(dW)
            db = dz # db = dz * 1
            grads_b.append(db)
            dA = np.dot(interspace.weights.T, dz) # prev_dA = T(W) . dz
            i -= 1
        
        return dict([
            ('weights', grads_W),
            ('biases', grads_b)
        ])

    def apply_gradient_descent(self, learning_rate: float, dW: list[np.ndarray], db: list[np.ndarray]) -> None:
        """
        Apply the gradient descent to improve the neural network
        """
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

def test_neural_network():
    nn = NeuralNetwork(3, 4, 2)
    X = np.array([4, 3, 1])
    y = np.array([0.5, 0.8])
    epochs = 10

    for _ in range(epochs):
        y_hat = nn.forward_prop(X)
        print(f"{y_hat=}")
        cost = NeuralNetwork.compute_cost(y, y_hat)
        print(f"{cost=}")
        grads = nn.back_prop(y)
        nn.apply_gradient_descent(0.1, grads['weights'], grads['biases'])

if __name__ == "__main__":
    # test_compute_cost()
    test_neural_network()
