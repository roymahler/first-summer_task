import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix


class DLLayer(object):
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2, optimization=None, regularization=None):
        # Constant parameters
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._learning_rate = learning_rate
        self._optimization = optimization
        self.alpha = learning_rate
        self.random_scale = 0.01
        self.init_weights(W_initialization)
        self.is_train = False
        self.regularization = regularization
        self.L2_lambda = 0
        self.dropout_keep_prob = 1

        # Optimization parameters
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *self._input_shape), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5

        # Regularization
        if self.regularization == "L2":
            self.L2_lambda = 0.6
        elif self.regularization == "dropout":
            self.dropout_keep_prob = 0.6

        # Activation
        self.activation_trim = 1e-10

        if self._activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward

        elif self._activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward

        elif self._activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward

        elif self._activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward

        elif self._activation == "relu":
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward

        elif self._activation == "leaky_relu":
            self.leaky_relu_d = 0.01
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward

        elif self._activation == "softmax":
            self.activation_forward = self._softmax
            self.activation_backward = self._softmax_backward

        elif self._activation == "trim_softmax":
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backward

        elif self._activation == "NoActivation":
            self.activation_forward = self._NoActivation
            self.activation_backward = self._NoActivation_backward

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization == "zeros":
            self.W = np.full((self._num_units, *self._input_shape), self.alpha)
        elif W_initialization == "random":
            self.W = np.random.randn(*(self._num_units, *self._input_shape)) * self.random_scale
        elif W_initialization == "Xaviar":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(1 / prev_l)
        elif W_initialization == "He":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(2 / prev_l)
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except FileNotFoundError:
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def set_train(self, is_train):
        self.is_train = is_train

    def set_dropout_keep_prob(self, prob):
        self.dropout_keep_prob = prob

    def set_L2_lambda(self, lmbda):
        self.L2_lambda = lmbda

    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = A = 1 / (1 + np.exp(-Z))
            TRIM = self.activation_trim
            if TRIM > 0:
                A = np.where(A < TRIM, TRIM, A)
                A = np.where(A > 1 - TRIM, 1 - TRIM, A)
            return A

    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self._Z)

        return dA * A * (1 - A)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        return dA * (1 - A ** 2)

    def _trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _trim_tanh_backward(self, dA):
        A = self._trim_tanh(self._Z)

        return dA * (1 - A ** 2)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_backward(self, dA):
        return np.where(self._Z <= 0, 0, dA)

    def _leaky_relu(self, Z):
        return np.where(Z <= 0, Z * self.leaky_relu_d, Z)

    def _leaky_relu_backward(self, dA):
        return np.where(self._Z <= 0, dA * self.leaky_relu_d, dA)

    def _softmax(self, Z):
        eZ = np.exp(Z)
        A = eZ / np.sum(eZ, axis=0)
        return A

    def _softmax_backward(self, dZ):
        return dZ

    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100, Z)
                eZ = np.exp(Z)
        A = eZ / np.sum(eZ, axis=0)
        return A

    def forward_propagation(self, A_prev, is_predict):
        if self.is_train and self.regularization == "dropout":
            self._A_prev = self.forward_dropout(A_prev)
        else:
            self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, self._A_prev) + self.b
        A = self.activation_forward(self._Z)
        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.dW = (1.0 / m) * np.dot(dZ, self._A_prev.T)
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        if self.regularization == "L2":
            m_L2 = dZ.shape[-1]
            self.dW += (self.L2_lambda/m_L2) * self.W
        dA_prev = np.dot(self.W.T, dZ)
        if self.regularization == "dropout":
            dA_prev = (dA_prev * self._D) / self.dropout_keep_prob
        return dA_prev

    def forward_dropout(self, A_prev):
        self._D = np.random.rand(*A_prev.shape)
        self._D = np.where(self._D > self.dropout_keep_prob, 0, 1)
        A_prev = (A_prev * self._D) / self.dropout_keep_prob
        return A_prev

    def _NoActivation(self, Z):
        return Z

    def _NoActivation_backward(self, dZ):
        return dZ

    def update_parameters(self):
        if self._optimization == "adaptive":
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont, self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont, self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b
        else:
            self.W -= self._learning_rate * self.dW
            self.b -= self._learning_rate * self.db

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(f"{path}/{file_name}.h5", 'w') as hf:
            hf.create_dataset('W', data=self.W)
            hf.create_dataset('b', data=self.b)

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"

        # optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"

        # parameters
        s += "\tparameters:\n\t\tW shape: " + str(self.W.shape) + "\n"
        s += "\t\tb shape: " + str(self.b.shape) + "\n"

        s += "\tactivation: " + self._activation + "\n"

        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d) + "\n"

        #plt.hist(self.W.reshape(-1))
        #plt.title("W histogram")
        #plt.show()

        # regulation
        s += "\tregulation: " + str(self.regularization) + "\n"
        if self.regularization == "L2":
            s += "\t\tL2 parameters:\n"
            s += "\t\t\tlambda: " + str(self.L2_lambda) + "\n"
        elif self.regularization == "dropout":
            s += "\t\tdropout parameters:\n"
            s += "\t\t\tkeep prob: " + str(self.dropout_keep_prob) + "\n"
        return s


class DLModel(object):
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        self.is_train = False
        self.inject_str_func = None

    def add(self, layer):
        self.layers.append(layer)

    def set_train(self, is_train):
        self.is_train = is_train
        for i in range(1, len(self.layers)):
            self.layers[i].set_train(is_train)

    def _squared_means(self, AL, Y):
        return (AL - Y) ** 2

    def _squared_means_backward(self, AL, Y):
        return 2 * (AL - Y)

    def _cross_entropy(self, AL, Y):
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def _cross_entropy_backward(self, AL, Y):
        dAL = np.where(Y == 0, 1 / (1 - AL), -1 / AL)
        return dAL

    def _categorical_cross_entropy(self, AL, Y):
        errors = np.where(Y == 1, -np.log(AL), 0)
        return errors

    def _categorical_cross_entropy_backward(self, AL, Y):
        dA = AL - Y
        return dA

    def compile(self, loss, threshold=0.5):
        self.threshold = threshold
        self.loss = loss.lower()

        if "squared" in loss and "means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif "categorical" in loss and "cross" in loss and "entropy" in loss:
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        elif "cross" in loss and "entropy" in loss:
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward
        self._is_compiled = True

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)
        sum = 0
        lambd = 0
        for i in range(1, len(self.layers)):
            if self.layers[i].L2_lambda != 0:
                sum += np.sum(np.square(self.layers[i].W))
                lambd = self.layers[i].L2_lambda
        return (np.sum(errors) / m) + (sum*lambd)/(2*m)

    def train(self, X, Y, num_iterations):
        self.set_train(True)
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
            # forward propagation
            Al = np.array(X, copy=True)
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)
			#backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1,L)):
                dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                self.layers[l].update_parameters()
			#record progress
            if num_iterations == 1 or (i > 0 and i % print_ind == 0):
                J = self.compute_cost(Al, Y)
                costs.append(J)
                inject_string = ""
                if self.inject_str_func != None:
                    inject_string = self.inject_str_func(self, X, Y, Y_hat)
                print("cost after ", str(i), "updates ("+str(i//print_ind)+"%):",str(J) + inject_string)
            else:
                if i > 0 and i % print_ind == 0:
                    J = self.compute_cost(Al, Y)
                    costs.append(J)
                    print("cost after ", str(i), "updates ("+str(i//print_ind)+"%):",str(J) + inject_string)
        self.set_train(True)
        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al, True)

        if Al.shape[0] > 1:
            return np.where(Al == Al.max(axis=0), 1, 0)
        return Al > self.threshold

    def confusion_matrix(self, X, Y):
        AL = self.predict(X)
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(Y, axis=0)
        return confusion_matrix(predictions, labels)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(1, len(self.layers)):
            self.layers[i].save_weights(path, f"Layer{i}")

    def _str_(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"

        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"

        return s
    
    def forward_propagation(self, X):
        L = len(self.layers)
        for l in range(1,L):
            X = self.layers[l].forward_propagation(X, False)            
        return X

    def backward_propagation(self, Al, Y):
        L = len(self.layers)
        dAl_t = self.loss_backward(Al, Y)                       # Starts with backwording the los function
        for l in reversed(range(1,L)):                          # Walk the ANN from the end to the begin (backword...)
            dAl_t = self.layers[l].backward_propagation(dAl_t)  # Backword
            self.layers[l].update_parameters()                  # Update parameters
        return dAl_t
