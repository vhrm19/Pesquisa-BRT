import numpy as np
import csv

dataset = []
data = csv.reader(open("csv desembarque.csv","r"), delimiter=';')
for line in data:
    line = [float(elemento) for elemento in line]
    dataset.append(line)
Tempo_por_Passageiro, Passageiros, Em_pe = [], [], []
for i in range(len(dataset)):
    Tempo_por_Passageiro.append([dataset[i][0]])
    Passageiros.append([dataset[i][1]])
    Em_pe.append([dataset[i][2]])
Entrada = np.column_stack((Passageiros,Em_pe))
Tempo_por_Passageiro = np.array(Tempo_por_Passageiro)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1 * (x > 0)

def bias(x):
    return np.column_stack((x, np.zeros(len(Entrada)).T))

# Forward Propagation
Entrada = bias(Entrada)
W1 = np.array([[np.random.random_sample(), np.random.random_sample(), np.random.random_sample()],
[np.random.random_sample(), np.random.random_sample(), np.random.random_sample()],
[0.1, 0.1, 0.1]])
Z2 = np.dot(Entrada, W1)
a2 = bias(ReLU(Z2))
W2 = np.array([[np.random.random_sample(), np.random.random_sample()],
[np.random.random_sample(), np.random.random_sample()],
[np.random.random_sample(), np.random.random_sample()],
[0.1, 0.1]])
Z3 = np.dot(a2, W2)
a3 = bias(ReLU(Z3))
W3 = np.array([[np.random.random_sample()], [np.random.random_sample()], [0.1]])
Z4 = np.dot(a3, W3)
a4 = ReLU(Z4)

# Gradient descent
d4 = (Tempo_por_Passageiro - a4) * -dReLU(Z4)
Grad3 = np.dot(a3.T, d4)
d3 = np.dot(d4, W3.T) * dReLU(Z2)
Grad2 = np.dot(a2.T, np.delete(d3, 2, 1))
d2 = np.dot(np.delete(d3, 2, 1), W2.T) * bias(dReLU(Z2))
Grad1 = np.dot(Entrada.T, np.delete(d2, 3, 1))

# Backpropagation
W1 = Grad1
Z2 = np.dot(Entrada, W1)
a2 = bias(ReLU(Z2))
W2 = Grad2
Z3 = np.dot(a2, W2)
a3 = bias(ReLU(Z3))
W3 = Grad3
Z4 = np.dot(a3, W3)
a4 = ReLU(Z4)

print(a4)