import numpy as np
import csv

def Initial_Weights(input, layers): 
    W = []
    S = [] # Squared gradient
    V = []
    S_hat = []
    V_hat = []
    for i in range(len(layers)):
        if i == 0:
            W.append(np.random.rand(len(input.T), layers[i])) # Gera pesos aleatorios de acordo com o shape da NN
        else:
            W.append(np.random.rand(layers[i-1], layers[i]))
        W[i] = np.vstack((W[i], np.full(len(W[i].T), 0.1))) # Ultimo número determina o bias
    for i in range(len(W)):
        S.append(np.zeros_like(W[i]))
        S_hat.append(np.zeros_like(W[i]))
        V.append(np.zeros_like(W[i]))
        V_hat.append(np.zeros_like(W[i]))
    return W, S, V, S_hat, V_hat

def Forward_Propagation(input, layers, W):
    Z = []
    a = []
    a.append(input) # Adiciona uma coluna de zeros na entrada
    for i in range(len(layers)):
        a[i] = np.column_stack((a[i], np.ones(len(a[i])).T))
        Z.append(np.dot(a[i], W[i])) # Calcula a entrada * pesos
        a.append(sigma(Z[i])) # Aplica a função de ativação
    return a, Z 

def Backpropagation(y, A, Z, w):
    Grad = []
    d = []
    for i in range(len(W)): # Calcula as derivadas parciais (gradientes) referentes a cada layer
        if i == 0:
            d.append(np.multiply(y - A[-1], -dsigma(Z[-i+len(W)-1])))
        else:
            d.append(np.multiply(np.dot(d[i-1], np.delete(w[-i+len(w)].T, -1, 1)), (dsigma(Z[-i+len(W)-1]))))
        Grad.append(np.dot(A[-i+len(W)-1].T, d[i]) - lamb*w[-i+len(W)-1])
    return Grad[::-1]

def Predict(Passageiros, Cheio):
    entrada = np.column_stack((Passageiros, Cheio))
    Out, nd = Forward_Propagation(entrada, layers, W)
    print("Tempo por Passageiro:", float(Out[-1]))

def Padroniza(x):
    media = np.mean(x, 0)
    desv = np.std(x, 0)
    x = (x - media) / desv
    return x, desv[0], media[0]

def Gradient_checking(w, grad):
    h = 0.0001
    V2 = []
    V1 = []
    for i in range(len(w)):
        for j in range(len(w[i])):
            for k in range(len(w[i][j])):
                w[i][j][k] += h
                a, nd = Forward_Propagation(Entrada, layers, w)
                loss2 = sum(0.5 * (Tempo_por_Passageiro - a[-1])**2)
                w[i][j][k] -= 2*h
                a, nd = Forward_Propagation(Entrada, layers, w)
                loss1 = sum(0.5 * (Tempo_por_Passageiro - a[-1])**2)
                V2.append((loss2 - loss1) / (2 * h))
                w[i][j][k] += h
    for i in range(len(grad)):
        for j in range(len(grad[i])):
            for k in range(len(grad[i][j])):
                V1.append(grad[i][j][k])
    print("Gradient checking:", np.linalg.norm(np.array(V1) - np.array(V2).T) / np.linalg.norm(np.array(V1) + np.array(V2).T))

def Minimiza(Entrada, layers, W, S, V, V_hat, S_hat, inter):
    for j in range(inter): # Faz o Backpropagation minimizando a função custo: 0.5 * sum(y-ŷ)**2, de acordo com o shape da NN
        a, Z = Forward_Propagation(Entrada, layers, W)
        Grad = Backpropagation(Tempo_por_Passageiro, a, Z, W)
        for i in range(len(W)):
            V[i] = 0.9 * V[i] + 0.1 * Grad[i]
            S[i] = 0.999 * S[i] + 0.001 * Grad[i]**2
            V_hat[i] = V[i] / 0.1
            S_hat[i] = S[i] / 0.001
            W[i] -= (0.001 / (S_hat[i] + 10E-8)**0.5) * V_hat[i]
    print("Custo minimizado:", float(0.5 * sum(Tempo_por_Passageiro-a[-1])**2))
    return W, Grad

def Dataset():
    dataset = []
    data = csv.reader(open("csv embarque.csv","r"), delimiter=';')
    for line in data:
        line = [float(elemento) for elemento in line]
        dataset.append(line)
    dataset, desv, media = Padroniza(dataset)
    Tempo_por_Passageiro, Passageiros, Em_pe = [], [], []
    for i in range(len(dataset)):
        Tempo_por_Passageiro.append([dataset[i][0]])
        Passageiros.append([dataset[i][1]])
        Em_pe.append([dataset[i][2]])
    Entrada = np.column_stack((Passageiros,Em_pe))
    Tempo_por_Passageiro = np.array(Tempo_por_Passageiro)
    return Entrada, Tempo_por_Passageiro, desv, media

funcao = 0 # Determinar funcao de ativacao: 0 = ReLU, 1 = Tanh, 2 = Sigmoid

if funcao == 0:
    def sigma(x):
        return x * (x > 0) # ReLU
    def dsigma(x):
        return 1 * (x > 0)
elif funcao == 1:
    def sigma(x):
        return np.tanh(x) # Tanh
    def dsigma(x):
        return 1 - np.tanh(x)**2
elif funcao == 2:
    def sigma(x):
        return 1 / (1 + np.exp(-x)) # Sigmoid
    def dsigma(x):
        return np.exp(-x) / (1 + np.exp(-x))**2        

Entrada, Tempo_por_Passageiro, desv, media = Dataset()

layers = [3,2,1] # Layers com seus respectivos neuronios

lamb = 0 # Parametro lambda da Regularização L2

np.random.seed(0) # Pesos aleatórios iniciais

W, S, V, S_hat, V_hat = Initial_Weights(Entrada, layers)

W, Grad = Minimiza(Entrada, layers, W, S, V, V_hat, S_hat, inter = 1000)

Gradient_checking(W, Grad)

Predict(1,2) # Entrada sendo (Numero de Passageiros no Ponto, O quao cheio esta o onibus: pouco, medio ou muito [1,2 ou 3])
