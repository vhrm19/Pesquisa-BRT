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

def ReLU(x):
    return x * (x > 0)
def dReLU(x):
    return 1 * (x > 0)

layers = [3, 2, 1]
bias = np.zeros(3)

def NNFFP(input, layers, bias):
    for i in range(len(layers)):
        W = np.random.rand(len(input.T))
        Z = np.dot(input, W)
        a = ReLU(Z + bias[i])
    return a

a = NNFFP(Entrada, layers, bias)

print(a)