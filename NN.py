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

def bias(x, tam): # Adiciona uma coluna de zeros de linhas tamanho "tam" na entrada dada
    return np.column_stack((x, np.zeros(len(tam)).T))

layers = [3, 2, 1] # Layers 1, 2 e 3, com 3, 2 e 1(saida) neuronios respectivamente

lamb = 0.1 # Parametro lambda da Regularização L2

W = [] # Pesos aleatórios iniciais

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
    a.append(bias(input, input)) # Adiciona uma coluna de zeros na entrada
    for i in range(len(layers)):
        Z.append(np.dot(a[i], W[i])) # Calcula a entrada * pesos
        a.append(bias(ReLU(Z[i]), input)) # Aplica a função de ativação
    a[-1] = np.delete(a[-1], 1, 1) # Deleta a coluna extra do bias
    return a, Z

a, Z = Forward_Propagation(Entrada, layers, W)

def Backpropagation(y, A, z, w):
    Grad = []
    d = []
    for i in range(3): # Calcula as derivadas parciais (gradientes) referentes a cada layer
        if i == 0:
            d.append(np.multiply(y - A[3], -dReLU(Z[2])))
            Grad.append(np.dot(A[2].T, d[0]) - lamb*w[-i+2]) # 
        if i == 1:
            d.append(np.multiply(np.dot(d[i-1], w[-i+3].T), bias(dReLU(z[-i+2]), y)))
            Grad.append(np.dot(A[-i+2].T, np.delete(d[i], i, 1)) - lamb*w[-i+2])
        if i == 2:
            d.append(np.multiply(np.dot(np.delete(d[i-1], 2, 1), w[-i+3].T), bias(dReLU(z[-i+2]), y)))
            Grad.append(np.dot(A[-i+2].T, np.delete(d[i], i, 1)) - lamb*w[-i+2])
    return Grad

Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

for j in range(5000): # Faz o Backpropagation minimizando a função custo: 0.5 * sum(y-ŷ)**2
    for i in range(len(W)):
        W[i] -= (0.01/50)*Grad[-i+2]
    a, Z = Forward_Propagation(Entrada, layers, W)
    Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

def Predict(Passageiros, Cheio):
    entrada = np.column_stack((Passageiros, Cheio))
    Out, nd = Forward_Propagation(entrada, layers, W)
    print("Tempo por Passageiro:", float(Out[-1]))

# Entrada sendo (Numero de Passageiros no Ponto, O quao cheio esta o onibus: pouco, medio ou muito [1,2 ou 3])
Predict(1,2)
