import numpy as np
from mlp_numpy import MLP

def main():
    # MLP con 2 entradas, una capa oculta de 3 neuronas (ReLU), salida de 1 neurona (Sigmoid)
    mlp = MLP([2, 3, 1], ['relu', 'sigmoid'])
    X = np.array([[0.5, -1.2], [1.0, 0.3]])  # batch de 2 ejemplos
    print("Predicción:", mlp.predict(X))

if __name__ == "__main__":
    main()
