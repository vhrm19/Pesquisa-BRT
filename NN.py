import numpy as np
import csv

dataset = []
data = csv.reader(open("csv embarque.csv","r"), delimiter=';')
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
    return np.column_stack((x, np.full((len(x),1), 0))) # Ultimo número determina o bias

def CS(x):
    return np.column_stack((x, np.ones(len(x)).T))

layers = [2, 3, 1] # Layers com seus respectivos neuronios

lamb = 0.1 # Parametro lambda da Regularização L2

np.random.seed(0) # Pesos aleatórios iniciais

def Forward_Propagation_Initial(input, layers): 
    Z = []
    a = []
    W = []
    a.append(bias(input)) # Adiciona uma coluna de zeros na entrada
    for i in range(len(layers)):
        W.append(np.random.rand(len(a[i].T), layers[i])) # Gera pesos aleatorios de acordo com o shape da NN
        Z.append(np.dot(a[i], W[i])) # Calcula a entrada * pesos
        a.append(bias(ReLU(Z[i]))) # Aplica a função de ativação
    a[-1] = np.delete(a[-1], -1, 1) # Deleta a coluna extra do bias
    return a, Z, W 

a, Z, W = Forward_Propagation_Initial(Entrada, layers)

def Forward_Propagation(input, layers, W):
    Z = []
    a = []
    a.append(bias(input)) # Adiciona uma coluna de zeros na entrada
    for i in range(len(layers)):
        Z.append(np.dot(a[i], W[i])) # Calcula a entrada * pesos
        a.append(bias(ReLU(Z[i]))) # Aplica a função de ativação
    a[-1] = np.delete(a[-1], -1, 1) # Deleta a coluna extra do bias
    return a, Z 

def Backpropagation(y, A, z, w):
    Grad = []
    d = []
    for i in range(len(W)): # Calcula as derivadas parciais (gradientes) referentes a cada layer
        if i == 0:
            d.append(np.multiply(y - A[-1], -dReLU(Z[-i+len(W)-1])))
        else:
            d.append(np.multiply(np.dot(d[i-1], w[-i+len(w)].T), CS(dReLU(z[-i+len(W)-1]))))
            d[i] = np.delete(d[i], -1, 1)
        Grad.append(np.dot(A[-i+len(W)-1].T, d[i]) - lamb*w[-i+len(W)-1])
    return Grad

Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

for j in range(len(layers) * 1000): # Faz o Backpropagation minimizando a função custo: 0.5 * sum(y-ŷ)**2, de acordo com o shape da NN
    for i in range(len(W)):
        W[i] -= (1 * 10**(-len(layers)) / len(Entrada)) * Grad[-i+len(W)-1]
    a, Z = Forward_Propagation(Entrada, layers, W)
    Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

def Predict(Passageiros, Cheio):
    entrada = np.column_stack((Passageiros, Cheio))
    Out, nd = Forward_Propagation(entrada, layers, W)
    print("Tempo por Passageiro:", float(Out[-1]))

# Entrada sendo (Numero de Passageiros no Ponto, O quao cheio esta o onibus: pouco, medio ou muito [1,2 ou 3])
Predict(1,2)
