from math import exp

import numpy as np


class Sigmoid:
    """
    Implements the Sigmoid activation function.

    The sigmoid function is defined as: f(x) = 1 / (1 + e^{-x})

    Attributes:
        input_layer: The layer that provides the input to this activation function.

    Methods:
        forward(): Applies the Sigmoid activation function to the output of the input layer.
        backward(downstream): Computes the gradient of the loss with respect to the input, which is then passed back to the previous layers.
    """

    @staticmethod
    def forward(input_array):
        # Apply the Sigmoid activation function to the output of the input layer
        output_array = 1 / (1 + np.exp(-input_array))
        return output_array

    @staticmethod
    def backward(downstream, input_array=None):
        # Compute the gradient of the loss with respect to the input
        input_grad = downstream * input_array * (1 - input_array)
        return input_grad
