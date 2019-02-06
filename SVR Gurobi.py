import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gurobipy import *

class SVR:
        def __init__(self, C = 0.1, e = 0.1):
                self.C = C
                self.e = e
                
        def fit(self, Input, Output):
                self.Input_std = Input
                self.Input = ((Input - np.mean(Input, 0)) / np.std(Input, 0)).tolist()
                self.Output = Output.tolist()
                ksi = []
                ksi_star = []

                model = Model()

                W = [   model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W0"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W1") ]
                b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bias")

                for i in range(len(self.Input)):
                        ksi.append(model.addVar(name=("ksi%d" % i)))
                        ksi_star.append(model.addVar(name=("ksi_star%d" % i)))

                obj = 0.5 * (W[0]*W[0] + W[1]*W[1]) + self.C*(quicksum(ksi) + quicksum(ksi_star))

                model.setObjective(obj, GRB.MINIMIZE)

                for i in range(len(self.Input)):
                        model.addConstr(self.Output[i][0] - (W[0]*self.Input[i][0] + W[1]*self.Input[i][1]) - b <= self.e + ksi[i] , name=("c1%d" %i))
                        model.addConstr(W[0]*self.Input[i][0] + W[1]*self.Input[i][1] + b - self.Output[i][0] <= self.e + ksi_star[i], name=("c2%d" %i))
                        model.addConstr(ksi[i] and ksi_star[i] >= 0, name=("c3%d" %i))

                model.update()
                model.optimize()

                results = {}
                if model.Status == GRB.OPTIMAL:
                        results['W'] = [W[0].X, W[1].X]
                        results['bias'] = b.X
                self.W = [W[0].X, W[1].X, b.X]
                print(results)

        def Graphs(self):
                x = []
                y = []
                z = []

                for i in range(1,10):
                        for j in range(1,4):
                                k = self.W[0]*i + self.W[1]*j + self.W[2]
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
