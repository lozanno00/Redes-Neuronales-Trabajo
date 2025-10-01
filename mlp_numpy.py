import numpy as np

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Neurona individual
def neuron(inputs, weights, bias, activation='sigmoid'):
    z = np.dot(inputs, weights) + bias
    if activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'relu':
        return relu(z)
    else:
        raise ValueError("Función de activación no soportada.")

# Capa de neuronas
class Layer:
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.activation = activation

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        else:
            raise ValueError("Función de activación no soportada.")

# MLP simple
class MLP:
    def __init__(self, layer_dims, activations):
        assert len(layer_dims) - 1 == len(activations)
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(Layer(layer_dims[i], layer_dims[i+1], activations[i]))

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

# Ejemplo de uso:
if __name__ == "__main__":
    # MLP con 2 entradas, una capa oculta de 3 neuronas (ReLU), salida de 1 neurona (Sigmoid)
    mlp = MLP([2, 3, 1], ['relu', 'sigmoid'])
    X = np.array([[0.5, -1.2], [1.0, 0.3]])  # batch de 2 ejemplos
    print("Predicción:", mlp.predict(X))
