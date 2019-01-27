import numpy as np
import csv
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self, Input, Output, layers = [3,2,1], bias = 1, rate = 1000):
        self.layers = layers
        self.Input = (Input - np.mean(Input, 0)) / np.std(Input, 0)
        self.Output = Output
        self.bias = bias
        self.rate = rate
        
    def Random_Weights(self):
        W = []
        for i in range(len(self.layers)):
            if i == 0:
                W.append(np.random.rand(len(self.Input.T), self.layers[i]))
            else:
                W.append(np.random.rand(self.layers[i-1], self.layers[i]))
            W[i] = np.vstack((W[i], np.full(len(W[i].T), self.bias)))
        return W

    def sigma(self, x):
        return x * (x > 0) # ReLU
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
        return a

        
    def MSE(self, W):
        y_hat = self.Forward_Propagation(self.Input, W)
        return 1/ len(self.Output) * sum((self.Output - y_hat[-1])**2)

    def GWO(self):
        x = []
        y = []

        Alpha = self.Random_Weights()
        Alpha_score=float("inf")
        Beta = self.Random_Weights()
        Beta_score=float("inf")
        Delta = self.Random_Weights()
        Delta_score=float("inf")
        W = self.Random_Weights()

        for i in range(self.rate):
            for j in range(len(self.layers)):
                fitness = self.MSE(W)

                if fitness < Alpha_score:
                    Alpha_score = fitness
                    Alpha = W
                if fitness > Alpha_score and fitness < Beta_score:
                    Beta_score = fitness
                    Beta = W
                if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                    Delta_score = fitness
                    Delta = W

            a = 2 -i *((2) /self.rate)

            for k in range(len(self.layers)):
                r1= np.random.random()
                r2= np.random.random()

                A1= 2* a* r1 -a
                C1= 2* r2

                D_alpha = abs(C1 * Alpha[k] - W[k])
                X1 = Alpha[k] - A1 * D_alpha

                r1= np.random.random()
                r2= np.random.random()

                A2= 2* a* r1 -a
                C2= 2* r2

                D_beta = abs(C2 * Beta[k] - W[k])
                X2 = Beta[k] - A2 * D_beta

                r1= np.random.random()
                r2= np.random.random()

                A3= 2* a* r1 -a
                C3= 2* r2

                D_delta = abs(C3 * Delta[k] - W[k])
                X3 = Delta[k] - A3 * D_delta

                W[k] = (X1+X2+X3)/3

            x.append(self.MSE(W))
            y.append(i)
        
        print("Cost:", self.MSE(W))
        plt.plot(y, x)

        return W

    def Predict(self, Passageiros, Cheio):
        W = self.GWO()
        entrada = np.column_stack((Passageiros, Cheio))
        Out = self.Forward_Propagation(entrada, W)
        print("Tempo por Passageiro:", float(Out[-1]))       

if __name__ == "__main__":

    np.random.seed(0)

    dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    NN = Neural_Network(Entrada, Tempo_por_Passageiro)
    NN.GWO()

    plt.ylabel("Custo")
    plt.xlabel("Iteração")
    plt.show()
