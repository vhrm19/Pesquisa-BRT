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
    return np.tanh(x)

def dReLU(x):
    return 1 - np.tanh(x)**2

layers = [3, 2, 1]

def Forward_Propagation(input, layers):
    W = []
    Z = []
    a = []
    a.append(input)
    for i in range(len(layers)):
        W.append(np.random.randn(len(a[i].T), layers[i]))
        Z.append(np.dot(a[i], W[i]))
        a.append(ReLU(Z[i]))
    return a, W, Z

def FP_withW(input, layers, W):
    Z = []
    a = []
    a.append(input)
    for i in range(len(layers)):
        Z.append(np.dot(a[i], W[i]))
        a.append(ReLU(Z[i]))
    return a

a, W, Z = Forward_Propagation(Entrada, layers)
Cost = 0.5 * sum(Tempo_por_Passageiro - a[3])**2

def Backpropagation(y, a, Z, W):
    Grad = []
    d = []
    for i in range(3):
        if i == 0:
            d.append(np.multiply(y - a[3], -dReLU(Z[2])))
        if i != 0:
            d.append(np.multiply(np.dot(d[i-1], W[-i+3].T), dReLU(Z[-i+2])))
        Grad.append(np.dot(a[-i+2].T, d[i]))
    return Grad

Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)
for i in range(len(W)):
    W[i] = W[i] - (0.01/50)*Grad[-i+2]

def Gradient_checking(W, Grad):
    Vpos = W
    Vneg = W
    loss2 = []
    loss1 = []
    h = 0.0001
    V1 = []
    for i in range(len(Grad)):
        for j in range(len(Grad[i])):
            for k in range(len(Grad[i][j])):
                V1.append([Grad[i][j][k]])
    for i in range(len(W)):
        for j in range(len(W[i])):
            Vpos = W
            Vneg = W
            for k in range(len(W[i][j])):
                Vpos[i][j][k] = W[i][j][k] + h
                a = FP_withW(Entrada, layers, Vpos)
                loss2.append(0.5 * sum(Tempo_por_Passageiro - a[3])**2)
                Vneg[i][j][k] = W[i][j][k] - h
                a = FP_withW(Entrada, layers, Vneg)
                loss1.append(0.5 * sum(Tempo_por_Passageiro - a[3])**2)
    V2 = (np.array(loss2) - np.array(loss1)) / (2 * h)
    return (V1 - V2) / (V1 + V2)

Check = Gradient_checking(W, Grad)
print(Check)