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
        Agents = 5

        y = []
        x = []
        W = []
        
        W_fit = np.zeros(Agents)
        Alpha = np.zeros_like(self.Random_Weights())
        Alpha_score=float("inf")
        Beta = np.zeros_like(self.Random_Weights())
        Beta_score=float("inf")
        Delta = np.zeros_like(self.Random_Weights())
        Delta_score=float("inf")

        for i in range(Agents):
            np.random.seed(i)
            W.append(self.Random_Weights())
            x.append([])
        
        for l in range(self.rate):
            
            a = 2 - l * 2 / self.rate

            for j in range(Agents):
                for k in range(len(self.layers)):

                    r1= np.random.random()
                    r2= np.random.random()

                    A1= 2* a* r1 -a
                    C1= 2* r2

                    D_alpha = abs(C1 * Alpha[k] - W[j][k])
                    X1 = Alpha[k] - A1 * D_alpha

                    r1= np.random.random()
                    r2= np.random.random()

                    A2= 2* a* r1 -a
                    C2= 2* r2

                    D_beta = abs(C2 * Beta[k] - W[j][k])
                    X2 = Beta[k] - A2 * D_beta

                    r1= np.random.random()
                    r2= np.random.random()

                    A3= 2* a* r1 -a
                    C3= 2* r2

                    D_delta = abs(C3 * Delta[k] - W[j][k])
                    X3 = Delta[k] - A3 * D_delta

                    W[j][k] = (X1+X2+X3)/3

            for i in range(Agents):
                W_fit[i] = self.MSE(W[i])

                x[i].append(float(W_fit[i]))
                
                if W_fit[i] < Alpha_score:
                    Alpha_score = W_fit[i]
                    Alpha = W[i].copy()
                if W_fit[i] > Alpha_score and W_fit[i] < Beta_score:
                    Beta_score = W_fit[i]
                    Beta = W[i].copy()
                if W_fit[i] > Alpha_score and W_fit[i] > Beta_score and W_fit[i] < Delta_score:
                    Delta_score = W_fit[i]
                    Delta = W[i].copy()
            y.append(l)

        for i in range(Agents):
            plt.plot(y, x[i], label = ('Agent', i+1))
        plt.plot(np.argmin(x)%self.rate, np.amin(x), 'b+')

        print('Cost:', float(Alpha_score))

        return Alpha

    def Predict(self, Passageiros, Cheio):
        W = self.GWO()
        entrada = np.column_stack((Passageiros, Cheio))
        Out = self.Forward_Propagation(entrada, W)
        print("Tempo por Passageiro:", float(Out[-1]))       

if __name__ == "__main__":
    
    dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    NN = Neural_Network(Entrada, Tempo_por_Passageiro)
    NN.Predict(1,2)

    plt.ylabel("Custo")
    plt.xlabel("Iteração")
    plt.legend()
    plt.show()
