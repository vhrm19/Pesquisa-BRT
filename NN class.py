import numpy as np
import csv

class Neural_Network:
    def __init__(self, Input, Output, layers, bias, rate):
        self.layers = layers
        self.Input = (Input - np.mean(Input, 0)) / np.std(Input, 0)
        self.Output = Output
        self.bias = bias
        self.lamb = 0
        self.rate = rate

    def Initial_Weights(self):
        W = []
        for i in range(len(self.layers)):
            if i == 0:
                W.append(np.random.rand(len(self.Input.T), self.layers[i]))
            else:
                W.append(np.random.rand(self.layers[i-1], self.layers[i]))
            W[i] = np.vstack((W[i], np.full(len(W[i].T), self.bias)))
        return W

    def sigma(self, x):
        return x * (x > 0)
    def dsigma(self, x):
        return 1 * (x > 0)

    def Forward_Propagation(self, _input, W):
        a = []
        Z = []
        a.append(_input)
        for i in range(len(self.layers)):
            a[i] = np.column_stack((a[i], np.ones(len(a[i])).T))
            Z.append(np.dot(a[i], W[i]))
            a.append(self.sigma(Z[i]))
        return a, Z

    def Backpropagation(self, a, Z, W):
        Grad = []
        d = []
        for i in range(len(W)):
            if i == 0:
                d.append(np.multiply(self.Output - a[-1], - self.dsigma(Z[-i + len(W) -1])))
            else:
                d.append(np.multiply(np.dot(d[i-1], np.delete(W[-i + len(W)].T, -1, 1)), (self.dsigma(Z[-i + len(W) -1]))))
            Grad.append(np.dot(a[-i + len(W) -1].T, d[i]) - self.lamb * W[-i + len(W) -1])
        return Grad[::-1]
        
    def Optimize(self):
        S = []
        V = []
        V_hat = []
        Cost = 1000000
        W = self.Initial_Weights()
        for i in range(len(W)):
            S.append(np.zeros_like(W[i]))
            V.append(np.zeros_like(W[i]))
            V_hat.append(np.zeros_like(W[i]))
        for i in range(self.rate):
            a, Z = self.Forward_Propagation(self.Input, W)
            Grad = self.Backpropagation(a, Z, W)
            for i in range(len(W)):
                V[i] = 0.9 * V[i] + 0.1 * Grad[i]
                V_hat[i] = V[i] / 0.1
                S[i] = np.maximum(0.999 * S[i], np.abs(Grad[i]))
                W[i] -= (0.002 / S[i]) * V_hat[i]
                Inst = float(0.5 * sum(self.Output-a[-1])**2)
                if Inst < Cost:
                    Cost = Inst
                    W_hat = W
                    Grad_hat = Grad
        print("Minimum:", Cost)
        return W_hat, Grad_hat

    def Predict(self, x1, x2):
        _input = np.column_stack((x1, x2))
        W, Grad = self.Optimize()
        print(Grad)
        self.Gradient_checking(W, Grad)
        Out, nd = self.Forward_Propagation(_input, W)
        print("Result:", float(Out[-1]))

    def Gradient_checking(self, w, grad):
        h = 0.0001
        V2 = []
        V1 = []
        for i in range(len(w)):
            for j in range(len(w[i])):
                for k in range(len(w[i][j])):
                    w[i][j][k] += h
                    a, nd = self.Forward_Propagation(self.Input, w)
                    loss2 = sum(0.5 * (self.Output - a[-1])**2)
                    w[i][j][k] -= 2*h
                    a, nd = self.Forward_Propagation(self.Input, w)
                    loss1 = sum(0.5 * (self.Output - a[-1])**2)
                    V2.append((loss2 - loss1) / (2 * h))
                    w[i][j][k] += h
        for i in range(len(grad)):
            for j in range(len(grad[i])):
                for k in range(len(grad[i][j])):
                    V1.append(grad[i][j][k])
        print("Gradient checking:", np.linalg.norm(np.array(V1) - np.array(V2).T) / np.linalg.norm(np.array(V1) + np.array(V2).T))

if __name__ == "__main__":

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

    np.random.seed(0)

    NN = Neural_Network(Entrada, Tempo_por_Passageiro, layers = [3,2,1], bias = 0, rate = 10000)
    
    NN.Predict(1,2)
