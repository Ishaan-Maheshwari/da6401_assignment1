import numpy as np

class Optimizer:
    def __init__(self, optimizer, learning_rate, momentum=0.5, beta=0.5, beta1=0.5, beta2=0.5, epsilon=1e-6, weight_decay=0.0):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 1  # Time step for Adam/Nadam

        # Variables for momentum-based methods
        self.velocity_w = None
        self.velocity_b = None
        self.squared_grad_w = None
        self.squared_grad_b = None
        self.m_w = None
        self.m_b = None

    def initialize(self, weights, biases):
        """Initialize optimizer-specific variables."""
        self.velocity_w = [np.zeros_like(w) for w in weights]
        self.velocity_b = [np.zeros_like(b) for b in biases]
        self.squared_grad_w = [np.zeros_like(w) for w in weights]
        self.squared_grad_b = [np.zeros_like(b) for b in biases]
        self.m_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, grads_w, grads_b):
        """Update weights and biases based on the selected optimizer."""
        if self.optimizer == 'sgd':
            self._sgd_update(weights, biases, grads_w, grads_b)
        elif self.optimizer == 'momentum':
            self._momentum_update(weights, biases, grads_w, grads_b)
        elif self.optimizer == 'nag':
            self._nag_update(weights, biases, grads_w, grads_b)
        elif self.optimizer == 'rmsprop':
            self._rmsprop_update(weights, biases, grads_w, grads_b)
        elif self.optimizer == 'adam':
            self._adam_update(weights, biases, grads_w, grads_b)
        elif self.optimizer == 'nadam':
            self._nadam_update(weights, biases, grads_w, grads_b)
        
        self.t += 1  # Increment time step

    def _sgd_update(self, weights, biases, grads_w, grads_b):
        """Stochastic Gradient Descent (SGD)"""
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * grads_w[i]
            biases[i] -= self.learning_rate * grads_b[i]

    def _momentum_update(self, weights, biases, grads_w, grads_b):
        """Momentum-based Gradient Descent"""
        for i in range(len(weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grads_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grads_b[i]
            weights[i] += self.velocity_w[i]
            biases[i] += self.velocity_b[i]

    def _nag_update(self, weights, biases, grads_w, grads_b):
        """Nesterov Accelerated Gradient (NAG)"""
        for i in range(len(weights)):
            prev_velocity_w = self.velocity_w[i].copy()
            prev_velocity_b = self.velocity_b[i].copy()
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grads_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grads_b[i]
            weights[i] += -self.momentum * prev_velocity_w + (1 + self.momentum) * self.velocity_w[i]
            biases[i] += -self.momentum * prev_velocity_b + (1 + self.momentum) * self.velocity_b[i]

    def _rmsprop_update(self, weights, biases, grads_w, grads_b):
        """RMSprop Update Rule"""
        for i in range(len(weights)):
            self.squared_grad_w[i] = self.beta * self.squared_grad_w[i] + (1 - self.beta) * grads_w[i] ** 2
            self.squared_grad_b[i] = self.beta * self.squared_grad_b[i] + (1 - self.beta) * grads_b[i] ** 2
            weights[i] -= self.learning_rate * grads_w[i] / (np.sqrt(self.squared_grad_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * grads_b[i] / (np.sqrt(self.squared_grad_b[i]) + self.epsilon)

    def _adam_update(self, weights, biases, grads_w, grads_b):
        """Adam Update Rule"""
        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.squared_grad_w[i] = self.beta2 * self.squared_grad_w[i] + (1 - self.beta2) * grads_w[i] ** 2
            self.squared_grad_b[i] = self.beta2 * self.squared_grad_b[i] + (1 - self.beta2) * grads_b[i] ** 2

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.squared_grad_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.squared_grad_b[i] / (1 - self.beta2 ** self.t)

            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def _nadam_update(self, weights, biases, grads_w, grads_b):
        """Nadam Update Rule (Adam + Nesterov Momentum)"""
        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.squared_grad_w[i] = self.beta2 * self.squared_grad_w[i] + (1 - self.beta2) * grads_w[i] ** 2
            self.squared_grad_b[i] = self.beta2 * self.squared_grad_b[i] + (1 - self.beta2) * grads_b[i] ** 2

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.squared_grad_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.squared_grad_b[i] / (1 - self.beta2 ** self.t)

            # Nadam modification (incorporating lookahead gradient)
            m_w_nadam = self.beta1 * m_w_hat + (1 - self.beta1) * grads_w[i] / (1 - self.beta1 ** self.t)
            m_b_nadam = self.beta1 * m_b_hat + (1 - self.beta1) * grads_b[i] / (1 - self.beta1 ** self.t)

            weights[i] -= self.learning_rate * m_w_nadam / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_nadam / (np.sqrt(v_b_hat) + self.epsilon)
