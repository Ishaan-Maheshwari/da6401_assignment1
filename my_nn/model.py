import numpy as np
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax
from .initializers import random_init, xavier_init
from .optimizers import Optimizer

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_neurons, output_size,
                 activation='sigmoid', weight_init='random', optimizer='sgd',
                 learning_rate=0.1, momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5, epsilon=1e-6, weight_decay=0.0, 
                 loss_function="cross_entropy"):

        # Select activation function
        activation_funcs = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "ReLU": (relu, relu_derivative),
            "identity": (lambda x: x, lambda x: np.ones_like(x))
        }
        if activation not in activation_funcs:
            raise ValueError("Invalid activation function")
        
        self.activation, self.activation_derivative = activation_funcs[activation]

        # Select weight initialization function
        self.weight_init = xavier_init if weight_init == 'Xavier' else random_init

        # Loss function selection
        self.loss_function = loss_function
        if loss_function not in ["cross_entropy", "mean_squared_error"]:
            raise ValueError("Invalid loss function")

        # Initialize model parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_size = output_size

        # Initialize weights and biases
        self.weights = [self.weight_init((input_size, hidden_neurons))]
        self.biases = [np.zeros((1, hidden_neurons))]

        for _ in range(hidden_layers - 1):
            self.weights.append(self.weight_init((hidden_neurons, hidden_neurons)))
            self.biases.append(np.zeros((1, hidden_neurons)))

        self.weights.append(self.weight_init((hidden_neurons, output_size)))
        self.biases.append(np.zeros((1, output_size)))

        # Initialize optimizer
        self.optimizer = Optimizer(optimizer, learning_rate, momentum, beta, beta1, beta2, epsilon, weight_decay)
        self.optimizer.initialize(self.weights, self.biases)

    def forward(self, X):
        """ Forward propagation through the network """
        activations = [X]
        raw_outputs = []

        for i in range(self.hidden_layers):
            raw_output = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activated_output = self.activation(raw_output)
            raw_outputs.append(raw_output)
            activations.append(activated_output)

        # Output layer (Softmax for classification)
        raw_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        activated_output = softmax(raw_output)
        raw_outputs.append(raw_output)
        activations.append(activated_output)

        return activations, raw_outputs

    def compute_loss(self, y_true, y_pred):
        """ Compute loss based on selected loss function """
        if self.loss_function == "cross_entropy":
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))  # Avoid log(0)
        elif self.loss_function == "mean_squared_error":
            return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, activations, raw_outputs):
        """ Backpropagation algorithm """
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Compute error at output layer
        y_pred = activations[-1]
        if self.loss_function == "cross_entropy":
            error = y_pred - y_true  # Softmax + cross-entropy derivative
        else:  # MSE
            error = (y_pred - y_true) * y_pred * (1 - y_pred)  # MSE derivative

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            gradients_w[i] = np.dot(activations[i].T, error) / X.shape[0]
            gradients_b[i] = np.sum(error, axis=0, keepdims=True) / X.shape[0]

            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.activation_derivative(activations[i])

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """ Apply optimizer updates to weights and biases. """
        self.optimizer.update(self.weights, self.biases, gradients_w, gradients_b)


    def train_step(self, X_batch, y_batch):
        """ Perform one training step (forward + backward + update) """
        activations, raw_outputs = self.forward(X_batch)
        gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, raw_outputs)
        self.update_weights(gradients_w, gradients_b)
        loss = self.compute_loss(y_batch, activations[-1])
        return loss

    def train(self, X_train, y_train, batch_size):
        """ Train for one epoch with batch gradient descent """
        num_samples = X_train.shape[0]
        total_loss = 0

        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            batch_loss = self.train_step(X_batch, y_batch)
            total_loss += batch_loss * X_batch.shape[0]  # Weighted sum

        return total_loss / num_samples  # Average loss

    def evaluate(self, X, y_true):
        """ Evaluate model performance on given data """
        activations, _ = self.forward(X)
        y_pred = activations[-1]

        # Compute loss
        loss = self.compute_loss(y_true, y_pred)

        # Compute accuracy
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)

        return loss, accuracy

    def predict(self, X):
        """ Generate predictions for input X """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)  # Return class label predictions
