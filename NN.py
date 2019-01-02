import numpy as np
import csv

def ReLU(x):
    if funcao == 0:
        return x * (x > 0) # ReLU
    if funcao == 1:
        return np.tanh(x) # Tanh
    if funcao == 2:
        return 1 / (1 + np.exp(-x)) # Sigmoid

def dReLU(x):
    if funcao == 0:
        return 1 * (x > 0)
    if funcao == 1:
        return 1 - np.tanh(x)**2
    if funcao == 2:
        return np.exp(-x) / (1 + np.exp(-x))**2

def bias(x):
    return np.vstack((x, np.full(len(x.T), 0))) # Ultimo número determina o bias

def CS(x):
    return np.column_stack((x, np.ones(len(x)).T))

def Initial_Weights(input, layers): 
    W = []
    for i in range(len(layers)):
        if i == 0:
            W.append(np.random.rand(len(input.T), layers[i])) # Gera pesos aleatorios de acordo com o shape da NN
        else:
            W.append(np.random.rand(layers[i-1], layers[i]))
        W[i] = bias(W[i])
    return W 

def Forward_Propagation(input, layers, W):
    Z = []
    a = []
    a.append(input) # Adiciona uma coluna de zeros na entrada
    for i in range(len(layers)):
        a[i] = CS(a[i])
        Z.append(np.dot(a[i], W[i])) # Calcula a entrada * pesos
        a.append(ReLU(Z[i])) # Aplica a função de ativação
    return a, Z 

def Backpropagation(y, A, Z, w):
    Grad = []
    d = []
    for i in range(len(W)): # Calcula as derivadas parciais (gradientes) referentes a cada layer
        if i == 0:
            d.append(np.multiply(y - A[-1], -dReLU(Z[-i+len(W)-1])))
        else:
            d.append(np.multiply(np.dot(d[i-1], np.delete(w[-i+len(w)].T, -1, 1)), (dReLU(Z[-i+len(W)-1]))))
        Grad.append(np.dot(A[-i+len(W)-1].T, d[i]) - lamb*w[-i+len(W)-1])
    return Grad

dataset = []
data = csv.reader(open("csv embarque padronizado.csv","r"), delimiter=';')
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

funcao = 0 # Determinar funcao de ativacao: 0 = ReLU, 1 = Tanh, 2 = Sigmoid

layers = [1] # Layers com seus respectivos neuronios

lamb = 0 # Parametro lambda da Regularização L2

np.random.seed(0) # Pesos aleatórios iniciais

W = Initial_Weights(Entrada, layers)

a, Z = Forward_Propagation(Entrada, layers, W)

Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

print("Custo nao minimizado:" , float(0.5 * sum(Tempo_por_Passageiro-a[-1])**2))

for j in range(len(layers) * sum(layers) * 1000): # Faz o Backpropagation minimizando a função custo: 0.5 * sum(y-ŷ)**2, de acordo com o shape da NN
    for i in range(len(W)):
        W[i] -= (10**(-len(layers)) / len(Entrada)) * Grad[-i+len(W)-1]
    a, Z = Forward_Propagation(Entrada, layers, W)
    Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)

print("Custo minimizado:" , float(0.5 * sum(Tempo_por_Passageiro-a[-1])**2))

def Predict(Passageiros, Cheio):
    entrada = np.column_stack((Passageiros, Cheio))
    Out, nd = Forward_Propagation(entrada, layers, W)
    print("Tempo por Passageiro:", float((Out[-1]*1.397990069)+2.0253333)) # (Out[-1]*2.679165479)+2.446792929 EMBARQUE; (Out[-1]*1.397990069)+2.0253333) DESEMBARQUE

# Entrada sendo (Numero de Passageiros no Ponto, O quao cheio esta o onibus: pouco, medio ou muito [1,2 ou 3])
Predict(1,2)
