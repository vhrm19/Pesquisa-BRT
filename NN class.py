import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Activation_functions:
    def __init__(self, func):
        self.func = func
    
    def sigma(self, x):
        if self.func == 0:
            x = x * (x > 0) # ReLU
            return x
        elif self.func == 1:
            x = np.tanh(x) # Tanh
            return x
        elif self.func == 2:
            x = 1 / (1 + np.exp(-x)) # Sigmoid
            return x
        elif self.func == 3:
            x = np.log(1 + np.exp(x)) # Softplus
            return x

    def dsigma(self, x):
        if self.func == 0:
            x = 1 * (x > 0)
            return x
        elif self.func == 1:
            x = 1 - np.tanh(x)**2
            return x
        elif self.func == 2:
            x = np.exp(-x) / (np.exp(-x) + 1)**2
            return x
        elif self.func == 3:
            x = 1 / (1 + np.exp(-x))
            return x

class Gradient_Optimisations:
    def __init__(self, func):
        self.func = func

    def Gradient_algorithm(self, W, Grad, V, S, S_hat, V_hat):
        if self.func == 0: # SGD
            for i in range(len(W)):
                W[i] -= 0.002 * Grad[i]
            return W, V, S, S_hat, V_hat 

        if self.func == 1: # Adam
            for i in range(len(W)):
                V[i] = 0.9 * V[i] + 0.1 * Grad[i]
                V_hat[i] = V[i] / 0.1
                S[i] = 0.999 * S[i] + 0.001 * Grad[i]**2
                S_hat[i] = S[i] / 0.001
                W[i] -= (0.001 / (S_hat[i] + 10E-8)**0.5) * V_hat[i]
            return W, V, S, S_hat, V_hat

        if self.func == 2: # RMSprop
            for i in range(len(W)):
                S[i] =  0.9 * S[i] + 0.1 * Grad[i]**2
                W[i] -= (0.001 / (S[i] + 10E-6)**0.5) * Grad[i]
            return W, V, S, S_hat, V_hat

        if self.func == 3: # AdaMax
            for i in range(len(W)):
                V[i] = 0.9 * V[i] + 0.1 * Grad[i]
                V_hat[i] = V[i] / 0.1
                S[i] = np.maximum(0.999 * S[i], np.abs(Grad[i]))
                W[i] -= (0.002 / S[i]) * V_hat[i]
            return W, V, S, S_hat, V_hat

        if self.func == 4: # AMSGrad
            for i in range(len(W)):
                V[i] = 0.9 * V[i] + 0.1 * Grad[i]
                S[i] = 0.999 * S[i] + 0.001 * Grad[i]**2
                S_hat[i] = np.maximum(S_hat[i], S[i])
                W[i] -= (0.001 / (S_hat[i] + 10E-8)**0.5) * V[i]
            return W, V, S, S_hat, V_hat

class Neural_Network:
    def __init__(self, Input, Output, layers = [3,2,1], bias = 0.1, rate = 1000, 
                    _lambda = 0.1, optimizer = 0, activation = 0, gradient_algorithm = 4, agents = 5):
        self.Agents = agents
        self.AC = Activation_functions(activation)
        self.GA = Gradient_Optimisations(gradient_algorithm)
        self.layers = layers
        self.Input_std = Input
        self.Input = (Input - np.mean(Input, 0)) / np.std(Input, 0)
        self.Output = Output
        self.bias = bias
        self._lambda = _lambda
        self.rate = rate
        if optimizer == 'Grad':
            self.W = self.Gradient_descent()
        elif optimizer == 'GWO':
            self.W = self.GWO()
        else:
            raise TypeError('Choose an optimizer: \'Grad\' for Gradient descent or \'GWO\' for Grey Wolf Optimizer')
        
    def Random_Weights(self):
        W = []
        for i in range(len(self.layers)):
            if i == 0:
                W.append(np.random.rand(len(self.Input.T), self.layers[i]))
            else:
                W.append(np.random.rand(self.layers[i-1], self.layers[i]))
            W[i] = np.vstack((W[i], np.full(len(W[i].T), self.bias)))
        return W 

    def Forward_Propagation(self, _input, W):
        a = []
        Z = []
        a.append(_input)
        for i in range(len(self.layers)):
            a[i] = np.column_stack((a[i], np.ones(len(a[i])).T))
            Z.append(np.dot(a[i], W[i]))
            a.append(self.AC.sigma(Z[i]))
        return a, Z

    def Backpropagation(self, a, Z, W):
        Grad = []
        d = []
        for i in range(len(W)):
            if i == 0:
                d.append(np.multiply(self.Output - a[-1], - self.AC.dsigma(Z[-i + len(W) -1])))
            else:
                d.append(np.multiply(np.dot(d[i-1], np.delete(W[-i + len(W)].T, -1, 1)), (self.AC.dsigma(Z[-i + len(W) -1]))))
            Grad.append(np.dot(a[-i + len(W) -1].T, d[i]) - self._lambda * W[-i + len(W) -1])
        return Grad[::-1]
        
    def Gradient_descent(self):
        x, y, z = [], [], []
        S = []
        V = []
        S_hat = []
        V_hat = []

        W = self.Random_Weights()

        for i in range(len(W)):
            S.append(np.zeros_like(W[i]))
            V.append(np.zeros_like(W[i]))
            S_hat.append(np.zeros_like(W[i]))
            V_hat.append(np.zeros_like(W[i]))

        Cost = 1000000

        print('Optimizing Neural Network with Gradient descent...')
        
        for k in range(len(W)):
            S.append(np.zeros_like(W[k]))
            V.append(np.zeros_like(W[k]))
            S_hat.append(np.zeros_like(W[k]))

        for j in range(self.rate):
            a, Z = self.Forward_Propagation(self.Input, W)
            Grad = self.Backpropagation(a, Z, W)

            W, V, S, S_hat, V_hat = self.GA.Gradient_algorithm(W, Grad, V, S, S_hat, V_hat)

            Check = self.Gradient_checking(W, Grad)
            Inst = self.MSE(W)

            if Inst < Cost:
                Cost = Inst
                W_hat = W
                Grad_hat = Grad
                step = j
                Check_min = Check
            
            x.append(Inst)
            y.append(j)
            z.append(Check)

        print("Step:", step, "Minimum:", float(Cost), "GC:", Check_min)
        plt.plot(step, Cost, 'b+')
        plt.plot(y, x, label  = "Cost")
        plt.plot(y, z, label  = "Grad. Checking")
        plt.ylabel("Cost")
        plt.xlabel("Step")
        plt.legend()

        return W_hat, Grad_hat

    def Predict(self, x1, x2, verbose = 1):
        _input = np.column_stack((x1, x2))
        W, grad = self.W
        Out, nd = self.Forward_Propagation(_input, W)
        if verbose != 0:
            print("Predicted result:", float(Out[-1]))
        return float(Out[-1])

    def Gradient_checking(self, w, grad):
        h = 0.0001
        V2 = []
        V1 = []
        for i in range(len(w)):
            for j in range(len(w[i])):
                for k in range(len(w[i][j])):
                    w[i][j][k] += h
                    loss2 = self.MSE(w)
                    w[i][j][k] -= 2*h
                    loss1 = self.MSE(w)
                    V2.append((loss2 - loss1) / (2 * h))
                    w[i][j][k] += h
        for i in range(len(grad)):
            for j in range(len(grad[i])):
                for k in range(len(grad[i][j])):
                    V1.append(grad[i][j][k])
        return np.linalg.norm(np.array(V1) - np.array(V2).T) / np.linalg.norm(np.array(V1) + np.array(V2).T)

    def Graphs(self):
        x = []
        y = []
        z = []

        for i in range(1,10):
            for j in range(1,4):
                k = NN.Predict(i,j, verbose = 0)
                x.append(i)
                y.append(j)
                z.append(k)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Y')
        ax.set_ylabel('X2')
        ax.set_xlabel('X1')
        ax.scatter(self.Input_std.T[0], self.Input_std.T[1], self.Output, c = 'red', label = 'Original Data')
        ax.plot_trisurf(x,y,z)
        plt.legend()
        plt.show()

    def Reshape(self, Weight):
        W = []
        shape = len(self.Input.T)
        start = 0
        for i in range(len(self.layers)):
            if i == 0:
                stop = (len(self.Input.T) + 1) * self.layers[0]
                W.append(np.reshape(Weight[start : stop], [shape + 1, self.layers[i]]))
            else:
                stop += (self.layers[i-1]+1) * self.layers[i]
                W.append(np.reshape(Weight[start : stop], [shape + 1, self.layers[i]]))
            shape = self.layers[i]
            start = stop
        return W
    
    def MSE(self, W):
        if len(W) != len(self.layers):
            W = self.Reshape(W)
        y_hat, nd = self.Forward_Propagation(self.Input, W)
        return 1/ len(self.Output) * sum((self.Output - y_hat[-1])**2)

    def GWO(self):

        print('Optimizing Neural Network with',self.Agents, 'Grey Wolfs...')

        y = []
        x = []
        W = []
        Pos = []

        for i in range(self.Agents):
            np.random.seed(i)
            W.append(self.Random_Weights())
            x.append([])
            for m in range(len(W[i])):
                for n in range(len(W[i][m])):
                    for o in range(len(W[i][m][n])):
                        Pos.append(W[i][m][n][o])
        Pos = np.split(np.array(Pos), self.Agents)
        
        W_fit = np.zeros(self.Agents)
        Alpha = np.zeros_like(Pos[0])
        Alpha_score=float("inf")
        Beta = np.zeros_like(Pos[0])
        Beta_score=float("inf")
        Delta = np.zeros_like(Pos[0])
        Delta_score=float("inf")

        for l in range(self.rate):

            for j in range(self.Agents):
                W_fit[j] = self.MSE(Pos[j])

                x[j].append(float(W_fit[j]))
                
                if W_fit[j] < Alpha_score:
                    Alpha_score = W_fit[j]
                    Alpha = Pos[j].copy()
                if W_fit[j] > Alpha_score and W_fit[j] < Beta_score:
                    Beta_score = W_fit[j]
                    Beta = Pos[j].copy()
                if W_fit[j] > Alpha_score and W_fit[j] > Beta_score and W_fit[j] < Delta_score:
                    Delta_score = W_fit[j]
                    Delta = Pos[j].copy()
                
            a = 2 - l * 2 / self.rate
                
            for j in range(self.Agents):
                for k in range(len(Pos[j])):

                    r1= np.random.random()
                    r2= np.random.random()

                    A1= 2* a* r1 -a
                    C1= 2* r2

                    D_alpha = abs(C1 * Alpha[k] - Pos[j][k])
                    X1 = Alpha[k] - A1 * D_alpha

                    r1= np.random.random()
                    r2= np.random.random()

                    A2= 2* a* r1 -a
                    C2= 2* r2

                    D_beta = abs(C2 * Beta[k] - Pos[j][k])
                    X2 = Beta[k] - A2 * D_beta

                    r1= np.random.random()
                    r2= np.random.random()

                    A3= 2* a* r1 -a
                    C3= 2* r2

                    D_delta = abs(C3 * Delta[k] - Pos[j][k])
                    X3 = Delta[k] - A3 * D_delta

                    Pos[j][k] = (X1+X2+X3)/3

            y.append(l)

        for i in range(self.Agents):
            plt.plot(y, x[i], label = ('Agent', i+1))
        plt.plot(np.argmin(x)%self.rate, np.amin(x), 'b+')
        plt.legend()

        print("Step:", np.argmin(x)%self.rate, "Minimum:", np.amin(x), 'Final Alpha Score:', float(Alpha_score))

        return self.Reshape(Alpha), 0

if __name__ == "__main__":

    np.random.seed(0)

    dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    # optimizer = 'Grad' for Gradient descent or 'GWO' for Grey Wolf Optimizer
    # activation: 0 = RelU, 1 = Tanh, 2 = Sigmoid, 3 = SoftPlus
    # gradient_algorithm: 0 = SGD, 1 = Adam, 2 = RMSprop, 3 = Adamax, 4 = AMSGrad
    # agents = Agents nº for GWO

    NN = Neural_Network(Entrada, Tempo_por_Passageiro, optimizer = "GWO", activation = 0, gradient_algorithm = 3, agents = 30)
    NN.Predict(1,2)
    NN.Graphs()
    
