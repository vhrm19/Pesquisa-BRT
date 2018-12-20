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

layers = [3, 2, 1]

W = []

W.append(np.array([
            [0.01, 0.05, 0.07],
            [0.2, 0.041, 0.11],
            [0, 0, 0] # Linha referente a bias/10
        ]))

W.append(np.array([
    [0.04, 0.78],
    [0.4, 0.45],
    [0.65, 0.23],
    [0, 0] # Linha referente a bias/10
]))

W.append(np.array([
    [0.04],
    [0.41],
    [0] # Linha referente a bias/10
]))

def Forward_Propagation(input, layers, W):
    Z = []
    a = []
    a.append(bias(input))
    for i in range(len(layers)):
        Z.append(np.dot(a[i], W[i]))
        a.append(bias(ReLU(Z[i])))
    a[-1] = np.delete(a[-1], 1, 1)
    return a, Z

a, Z = Forward_Propagation(Entrada, layers, W)

def Backpropagation(y, a, Z, W):
    Grad = []
    d = []
    for i in range(3):
        if i == 0:
            d.append(np.multiply(y - a[3], -dReLU(Z[2])))
            Grad.append(np.dot(a[2].T, d[0]))
        if i == 1:
            d.append(np.multiply(np.dot(d[i-1], W[-i+3].T), bias(dReLU(Z[-i+2]))))
            Grad.append(np.dot(a[-i+2].T, np.delete(d[i], i, 1)))
        if i == 2:
            d.append(np.multiply(np.dot(np.delete(d[i-1], 2, 1), W[-i+3].T), bias(dReLU(Z[-i+2]))))
            Grad.append(np.dot(a[-i+2].T, np.delete(d[i], i, 1)))
    return Grad

Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

for j in range(60000):
    for i in range(len(W)):
        W[i] -= (0.01/50)*Grad[-i+2]
    a, Z = Forward_Propagation(Entrada, layers, W)
    Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

def Gradient_checking(W, Grad):
    loss2 = []
    loss1 = []
    V1 = []
    h = 0.0001
    for i in range(len(Grad)):
        for j in range(len(Grad[i])):
            for k in range(len(Grad[i][j])):
                V1.append([Grad[i][j][k]])
    for i in range(len(W)):
        for j in range(len(W[i])):
            Vpos = W
            Vneg = W
            for k in range(len(W[i][j])):
                Vpos[i][j][k] += h
                a, Z = Forward_Propagation(Entrada, layers, Vpos)
                loss2.append(0.5 * sum(Tempo_por_Passageiro - a[-1])**2)
                Vneg[i][j][k] -= 2*h
                a, Z = Forward_Propagation(Entrada, layers, Vneg)
                loss1.append(0.5 * sum(Tempo_por_Passageiro - a[-1])**2)
    V2 = (np.array(loss2) - np.array(loss1)) / (2 * h)
    print(np.abs(V1-V2) / np.abs(V1+V2))


Gradient_checking(W, Grad)
