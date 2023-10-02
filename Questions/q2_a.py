from Model.layers.network import BaseNetwork
import numpy as np
import matplotlib.pyplot as plt
from Model.layers.input import InputLayer
from Model.layers.hidden import HiddenLayer
from Model.loss.square_loss import SquareLoss
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q2_b
from Model.evaluate.evaluate import evaluate_model

Number_of_iterations = 3000  # Experiment to pick your own number of ITERATIONS
Step_size = 0.01  # Experiment to pick your own STEP number


class Network(BaseNetwork):
    def __init__(self, data_layer):
        super().__init__()
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, 1)
        self.bias_layer1 = BiasLayer(self.hidden_layer1,"ReLU")
        self.output_layer1 = OutputLayer(self.bias_layer1, 1)
        self.set_output_layer(self.output_layer1)


# To get you started we built the network for you!! Please use the template file to finish answering the question

class Trainer:
    def __init__(self):
        pass

    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers.
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        network = Network(data_layer=data_layer)
        return network

    def net_setup(self, training_data):
        x, y = training_data
        # TODO: define input data layer
        self.data_layer = Data(x)
        # TODO: construct the network. you don't have to use define_network.
        self.network = self.define_network(self.data_layer)
        # TODO: use the appropriate loss function here
        self.loss_layer = SquareLoss(self.network.get_output_layer(), y)
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optimizer = SGDSolver(Step_size, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optimizer

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function

        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optimizer.step()
        return loss

    def train(self, num_iter):
        train_losses = []
        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range(num_iter):
            train_losses.append(self.train_step())
            print("iteration num " + str(i))

        # you have to return train_losses for the function
        return train_losses


def main(test=False):
    trainer = Trainer()
    data = q2_b()
    data_layer, network, loss_layer, optimizer = trainer.net_setup(data["train"])
    loss = trainer.train(Number_of_iterations)
    plt.plot(loss)
    plt.xlabel("number of epochs")
    plt.ylabel("loss of network")
    plt.show()

    test_data_x, test_data_y = data["test"]
    network.input_layer = InputLayer(Data(test_data_x))
    network.hidden_layer1.input_layer = network.input_layer

    y_predictions = network.output_layer1.forward()

    metrics = evaluate_model(test_data_y, y_predictions)
    print("METRICS:" + str(metrics))


if __name__ == "__main__":
    main()
    pass
