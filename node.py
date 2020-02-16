import numpy as np

class Connection:
    def __init__(self, weight):
        self.weight = weight
        self.deltaweight = 0

# General node class to be used for normal neural networks, NEAT, or RNNs
class Node:
    # Has members f_activation:   activation function
    #             d_f_activation: derivative of the activation function
    #             bias:           bias
    #             activation:     the activation of the neuron
    #             input:          the input, for back propagation reasons
    #             delta:          error signal, also for back propogation
    #             children:       a list of tuples containing the child node as
    #                                well as the weight to the node
    #             parents:        a list of tuples containing the parent node as
    #                                well as the weight to the node
    def __init__(self, f_activation, d_f_activation, random_limit):
        self.f_activation = f_activation
        self.d_f_activation = d_f_activation
        self.bias = (np.random.random() - 0.5) * random_limit
        self.activation = 0
        self.input = 0
        self.delta = 0
        self.deltabias = 0
        self.children = []
        self.parents = []
        self.random_limit = random_limit

    # actually create the children. This would go in __init__, but then it would
    # be impossible to create a neuron without already having a neuron
    def create_children(self, children):
        for child in children:
            weight = Connection((np.random.random() - 0.5) * self.random_limit)
            self.children.append((child, weight))
            child.parents.append((self, weight))

    # Finishes forward propogation on itself (applying activation function and
    # bias) and calculates its contribution to its children nodes
    def prop(self):
        if self.f_activation is not None:
            self.activation = self.f_activation(self.input + self.bias)
        else:
            self.activation = self.input
        if len(self.children) > 0:
            for child, weight in self.children:
                child.input += self.activation * weight.weight

    # d_error is the derivative of the cost function with respect to the
    # activation of the output neuron
    def backprop(self, is_last_layer, d_error, training_rate):
        # self.delta is the chain rule product up to the given neuron
        if is_last_layer:
            self.delta = d_error * self.d_f_activation(self.input + self.bias)
        else:
            self.delta = 0
            for child in self.children:
                self.delta += child[0].delta * child[1].weight
            if self.d_f_activation is not None:
                self.delta *= self.d_f_activation(self.input + self.bias)
        if self.f_activation is not None:
            self.deltabias -= training_rate * self.delta
        for parent in self.parents:
            parent[1].deltaweight -= training_rate * parent[0].activation * self.delta
        self.input = 0
        self.activation = 0

    # update the weights and biase
    def update(self, batch_size):
        if self.f_activation is not None:
            self.bias += self.deltabias / batch_size
            for parent, weight in self.parents:
                weight.weight += weight.deltaweight / batch_size
                weight.deltaweight = 0
        self.deltabias = 0
