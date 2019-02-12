import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gurobipy import *
import sklearn as sk
from sklearn import svm

class Kernels:
        def __init__(self, opt, gamma, degree):
                self.opt = opt
                self.gamma = gamma
                self.degree = degree
        
        def Kernel(self, xi, xj):
                if self.opt == 0: # RBF
                        return np.exp(-self.gamma * np.linalg.norm(xi - xj)**2) 

                if self.opt == 1: # Polynomial
                        return pow(sum([xi[i] * xj[i] for i in range(len(xi))]), self.degree)

                if self.opt == 2: # Linear
                        return np.dot(xi, xj)

class SVR:
        def __init__(self, kernel = 0):
                self.kernel = kernel

        def fit(self, Input, Output, verbose = 1):
                self.Input_std = Input.copy()
                self.Input = ((Input - np.mean(Input, 0)) / np.std(Input, 0))
                self.Output = Output.tolist()
                self.Hyperparameters()

                a = []
                a_star = []

                model = Model()
                model.Params.OutputFlag = verbose
             
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
                        model.addConstr( 0 <= a[i] <= self.C, name=("c1%d" %i))
                        model.addConstr( 0 <= a_star[i] <= self.C, name=("c1%d" %i))

                model.update()
                model.optimize()
                
                self.a = []
                self.a_star = []
                for i in range(len(self.Input)):
                        self.a.append(a[i].X)
                        self.a_star.append(a_star[i].X)

        def Predict(self, x1, x2):
                return sum((self.a[i] - self.a_star[i]) * self.Kernel(np.asarray(self.Input[i]), (x1,x2)) 
                                for i in range(len(self.Input)))

        def Graphs(self):
                x = []
                y = []
                z = []

                for i in range(1,10):
                        for j in range(1,4):
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

        def LOOCV(self):
                print('Calculating Leave-one-out cross validation')
                cost = 0
                Input = self.Input.copy()
                Output = self.Output.copy()
                for i in range(len(Input)):
                        self.fit(np.delete(Input, i, 0), np.delete(Output, i, 0), verbose = 0)
                        cost += (Output[i] - self.Predict(Input[i][0], Input[i][1]))**2
                print("LOOCV:", float(cost / len(Input)))

        def MSE(self):
                cost = 0
                for i in range(len(self.Input)):
                        cost += (self.Output[i] - self.Predict(self.Input[i][0], self.Input[i][1]))**2
                return cost / len(self.Input)

        def Hyperparameters(self):
                svr = svm.SVR()
                param = {'C':np.random.uniform(low=0.1, high=10, size=(50,)),'gamma':np.random.uniform(low=1e-7, high=0.1, size=(50,)),
                        'epsilon':np.random.uniform(low=0.1, high=1, size=(50,)), 'degree':np.random.randint(low=1, high=10, size=(10,))}
                rand = sk.model_selection.RandomizedSearchCV(svr, param, cv = len(self.Input))
                Out = np.reshape(self.Output, len(self.Output))
                rand.fit(self.Input, Out)
                rand.best_params_
                param = rand.best_params_
                self.C = param['C']
                self.e = param['epsilon']     
                self.Kernel = Kernels(self.kernel, param['gamma'], param['degree']).Kernel

if __name__ == '__main__':
        np.random.seed(0)

        dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
        Entrada = np.hsplit(dataset, (0,1))[2]
        Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

        # kenel: 0 = RBF, 1 = Polynomial, 2 = Linear

        Svr = SVR(kernel = 1)
        Svr.fit(Entrada, Tempo_por_Passageiro)
        Svr.Graphs()
        Svr.LOOCV()
