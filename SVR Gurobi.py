import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gurobipy import *

class Kernels:
        def __init__(self, opt, gamma, degree, constant):
                self.opt = opt
                self.gamma = gamma
                self.degree = degree
                self.constant = constant
        
        def Kernel(self, xi, xj):
                if self.opt == 0: # RBF
                        return np.exp(-self.gamma * np.linalg.norm(xi - xj)**2) 

                if self.opt == 1: # Polynomial
                        return pow(sum([xi[i] * xj[i] for i in range(len(xi))]) + self.constant, self.degree)

                if self.opt == 2: # Linear
                        return np.dot(xi, xj)

class SVR:
        def __init__(self, gamma = 0.1, degree = 3, constant = 0, C = 0.1, e = 0.1, kernel = 0):
                self.C = C
                self.e = e     
                self.Kernel = Kernels(kernel, gamma, degree, constant).Kernel

        def fit(self, Input, Output):
                self.Input_std = Input.copy()
                self.Input = ((Input - np.mean(Input, 0)) / np.std(Input, 0))
                self.Output = Output.tolist()
                a = []
                a_star = []

                model = Model()

                K = np.array([self.Kernel(np.asarray(self.Input[i]), np.asarray(self.Input[j]))
                        for j in range(self.Input_std.shape[0])
                        for i in range(self.Input_std.shape[0])]).reshape((self.Input_std.shape[0],self.Input_std.shape[0])).tolist()

                for i in range(len(self.Input)):
                        a.append(model.addVar(lb=0, name=("a")))
                        a_star.append(model.addVar(lb=0, name=("a_star")))

                obj = (-0.5* quicksum((a[i] - a_star[i]) * (a[j] - a_star[j]) * K[i][j] for i in range(len(self.Input)) for j in range(len(self.Input)))
                        - self.e * quicksum(a[i] + a_star[i] for i in range(len(self.Input))) + quicksum(self.Output[i][0] * (a[i] - a_star[i]) for i in range(len(self.Input))))

                model.setObjective(obj, GRB.MAXIMIZE)

                model.addConstr(quicksum(a[i] - a_star[i] for i in range(len(self.Input))) == 0, name=("c0"))
                for i in range(len(self.Input)):
                        model.addConstr( 0 <= a[i] and a_star[i]  <= self.C, name=("c1%d" %i))

                model.update()
                model.optimize()
                
                self.a = []
                self.a_star = []
                for i in range(len(self.Input)):
                        self.a.append(a[i].X)
                        self.a_star.append(a_star[i].X)

                self.W = sum((a[i].X - a_star[i].X) * np.asarray(self.Input[i]) for i in range(len(self.Input)))
                print(self.W)

        def Predict(self, x1, x2):
                return sum((self.a[i] - self.a_star[i]) * self.Kernel(np.asarray(self.Input[i]), (x1,x2)) for i in range(len(self.Input)))

        def Graphs(self):
                x = []
                y = []
                z = []

                for i in range(10):
                        for j in range(4):
                                k = self.Predict(i,j)
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

if __name__ == '__main__':
        np.random.seed(0)

        dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
        Entrada = np.hsplit(dataset, (0,1))[2]
        Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

        Svr = SVR()
        Svr.fit(Entrada, Tempo_por_Passageiro)
        Svr.Graphs()
