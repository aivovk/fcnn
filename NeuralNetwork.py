"""
Code based on:
http://neuralnetworksanddeeplearning.com/

Create and train (SGD) a fully connected network with any number of layers and
neurons in each layer.

Neurons are sigmoid with MSE cost.

"""

import random
import numpy as np
#def relu(z):
#def f(z):
#    return z * (z > 0)

#def relu_prime(z):
#def f_prime(z):
#    return 1.0 * (z > 0)

#def sigmoid(z):
def f(z):
    return 1.0 / (1.0 + np.exp(-z))

#def sigmoid_prime(z):
def f_prime(z):
    # np.exp(-z) / (1.0 + np.exp(-z)) ** 2
    return f(z) * (1 - f(z))

# MSE cost
def cost_prime(a, y):
    return a - y

class FullyConnectedNetwork():
    def __init__(self, layer_sizes, seed = 3):
        """
        layer_sizes is a list with the number of neurons in each layer
        the first layer is the input layer so doesn't have weights/biases
        number of weights in each neuron is the number of neurons in the previous layer
        e.g. biases[0] and weights[0] refers to the connection between layers 0 and 1
        """
        np.random.seed(seed)
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(num_neurons) for num_neurons in layer_sizes[1:]]
        self.weights = [np.random.randn(num_neurons, num_inputs)
                        for num_inputs, num_neurons in zip(layer_sizes[:-1], layer_sizes[1:])]

        # these will change shape based on number of training inputs (one below):
        # activations
        self.a = [np.zeros(num_neurons) for num_neurons in layer_sizes]
        # first layer of the following aren't used:
        # weighted input a = f(z)
        self.z = [np.zeros(num_neurons) for num_neurons in layer_sizes]
        # weighted input error d = dC/dz
        self.d = [np.zeros(num_neurons) for num_neurons in layer_sizes]
        
    def feedforward(self, X):
        """
        given an input X, find the output of the network
        in X each row is an individual input and the columns are pixel values
        """
        self.a[0] = X.T
        for i in range(1, self.num_layers):
            # biases not broadcasted automatically?
            self.z[i] = np.matmul(self.weights[i-1], self.a[i-1]) + self.biases[i-1][:,None]
            self.a[i] = f(self.z[i])
        return self.a[-1]

    def backpropagation(self, X, y):
        self.feedforward(X)

        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # convert y from digit to array form (e.g. 4 = [0,0,0,0,1,0,0,0,0,0])
        num_train = len(y)
        y_activation = np.zeros(self.a[-1].shape)
        y_index =np.array([(y[i],i) for i in range(num_train)])
        y_activation[tuple(y_index.T)] = 1
        
        self.d[-1] = cost_prime(self.a[-1], y_activation) * f_prime(self.z[-1])
        delta_b[-1] = np.mean(self.d[-1], axis = 1)
        delta_w[-1] = np.matmul(self.d[-1], self.a[-2].T)
        
        for i in range(self.num_layers-2, 0, -1):
            self.d[i] = np.dot(np.transpose(self.weights[i]), self.d[i+1])
            self.d[i] *= f_prime(self.z[i])
            delta_b[i-1] = np.mean(self.d[i], axis=1)
            delta_w[i-1] = np.matmul(self.d[i], self.a[i-1].T) / num_train

        return (delta_b, delta_w)
        
    def update(self, train_data, learning_rate):
        """
        update weights and biases for an arbitrary number of training inputs
        """
        X, y = train_data
        backprop_b, backprop_w = self.backpropagation(X, y)
        
        # TODO are these for loop assignments faster than genexpr for list?
        # e.g. self.biases = [b - l/m * d for b,d in zip(self.biases, delta_b)]
        for conn in range(self.num_layers - 1):
            self.biases[conn] -= learning_rate * backprop_b[conn]
            self.weights[conn] -= learning_rate * backprop_w[conn]

    def evaluate(self, data):
        """
        count the number of correct predictions
        """
        X, y = data
        return sum(np.argmax(self.feedforward(X), axis = 0) == y)
 
    def train_SGD(self,
                  train_data,
                  learning_rate,
                  batch_size,
                  num_epochs,
                  test_data = None):
        num_train = len(train_data[0])
        if test_data:
            num_test = len(test_data[0])

        X, y = train_data
        indices = np.arange(num_train)

        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            train_batches = [(X[batch_start:batch_start+batch_size], y[batch_start:batch_start+batch_size])
                             for batch_start in range(0, num_train, batch_size)]
            for train_batch in train_batches:
                self.update(train_batch, learning_rate)
                
            if test_data:
                print("Epoch {0}: {1}/{2}"
                      .format(epoch, self.evaluate(test_data), num_test))
            else:
                print("Epoch {0}".format(epoch))
            

if  __name__=="__main__":
    nn = FullyConnectedNetwork([2,3,1])
    print(nn.weights)
    x = np.array([0,1])
    print(nn.feedforward(np.array([0,1])))
    print(x)
    test = [(np.array([0,1]), 0), (np.array([3,4]), 1)]
    print(nn.evaluate(test))
                        

                        
