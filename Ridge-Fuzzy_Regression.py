import numpy as np
from gurobipy import *
import csv
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV

class Ridge_Fuzzy:

    def otimiza(self, y, x, size, h, plot = True):
        self.h = h

        n = len(y)

        model = Model("qTSQ-PLS-PM with fuzzy scheme")
        model.setParam("OutputFlag", 0)
        awL = {}
        awR = {}

        for i in range(size + 1):
            awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
            awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)
        
        self.rr = RidgeCV(alphas=np.arange(1e-3, 1, 0.009), normalize=True, cv = len(x)).fit(x, y)
        ac = self.rr.coef_

        model.setObjective(quicksum((float(np.dot(x[:, j], x[:, j].transpose()))
                                    * (awL[j + 1] + awR[j + 1]) * (awL[j + 1] + awR[j + 1]))
                                    for j in range(size))
                        + (awL[0] + awR[0]) * (awL[0] + awR[0]),
                        GRB.MINIMIZE)

        for i in range(n):
            model.addConstr(
                quicksum((ac[j] * x[i, j])
                        for j in range(size))
                - (1 - h) * (awL[0] + quicksum((abs(x[i, j]) * awL[j + 1])
                                            for j in range(size))) <= y[i])

        for i in range(n):
            model.addConstr(
                quicksum((ac[j] * x[i, j])
                        for j in range(size))
                + (1 - h) * (awR[0] + quicksum((abs(x[i, j]) * awR[j + 1])
                                            for j in range(size))) >= y[i])
        model.optimize()

        if plot:
            plt.plot(x,y, "o", markersize=2, label='Data')
            x = np.arange(1,15)
            plt.plot(x, eval(str("%f + %f*x" %(awL[0].x, (awL[1].x + ac[0])))), "--", label = "awL")
            plt.plot(x, eval(str("%f + %f*x" %(awR[0].x, (awR[1].x + ac[0])))), "--", label = "awR")
            plt.plot(x, eval(str("%f + %f*x" %(self.rr.intercept_, ac[0]))), "--", label = "aC")
            plt.legend()
            plt.show()

        return model, ac, awL, awR


    def Good_of_fitness(self, x, y, ac, awL, awR, size):
        ylow = []
        yhigh = []
        ymid = []

        for i in range(len(y)):
            ylow.append((0 - awL[0].x) + [(ac[j] - awL[j + 1].x) *
                                        x[i, j] for j in range(size)][0])
        
        for i in range(len(y)):
            yhigh.append((0 + awR[0].x) + [(ac[j] + awR[j + 1].x) *
                                        x[i, j] for j in range(size)][0])

        for i in range(len(y)):
            ymid.append((ylow[i] + yhigh[i]) / 2)

        SST = 0
        for i in range(len(y)):
            SST += (y[i] - ylow[i]) * (y[i] - ylow[i]) + \
                (yhigh[i] - y[i]) * (yhigh[i] - y[i])

        SSR = 0
        for i in range(len(y)):
            SSR += (ymid[i] - ylow[i]) * (ymid[i] - ylow[i]) + \
                (yhigh[i] - ymid[i]) * (yhigh[i] - ymid[i])

        IC = float(SSR / SST)

        AFS = sum(self.h * abs((ac + awR[1].x)*x - (ac + awL[1].x)*x))

        MFC = float(sum((ac*x) / abs((ac + awR[1].x)*x - (ac + awL[1].x)*x)))

        return IC, AFS, MFC


    def GOF_plot(self, x, y, size):
        h = 0
        IClist = []
        AFSlist = []
        MFClist = []
        hlist = []

        for i in range(0, 19):
            h += 0.05
            hlist.append(h)
            model, ac, awL, awR = self.otimiza(y, x, size, h, plot = False)
            gof = self.Good_of_fitness(x, y, ac, awL, awR, size)
            IClist.append(gof[0])
            AFSlist.append(gof[1])
            MFClist.append(gof[2])

        plt.xlabel('índice h')
        plt.plot(hlist, IClist, label = 'IC')
        plt.plot(hlist, AFSlist, label = 'AFS')
        plt.plot(hlist, MFClist, label = 'MFC')
        plt.legend()
        plt.show()

def Passageiros(x):
    Pouco = []
    Medio = []
    Muito = []

    for i in range(len(x)):
        if x[i][2] == 1:
            Pouco.append([x[i][0],x[i][1]])
        elif x[i][2] == 2:
            Medio.append([x[i][0],x[i][1]])
        elif x[i][2] == 3:
            Muito.append([x[i][0],x[i][1]])
    return np.array(Pouco), np.array(Medio), np.array(Muito)

if __name__ == "__main__":

    data = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
    
    dataset = Passageiros(data)[2] # Passageiros: 0 = Poucos, 1 = Medio, 2 = Muitos

    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    m = Ridge_Fuzzy()

    model, ac, awL, awR = m.otimiza(Tempo_por_Passageiro, Entrada, 1, 0.2)
    gof = m.Good_of_fitness(Entrada, Tempo_por_Passageiro, ac, awL, awR, 1)
    print("IC = %f, AFS = %f, MFC = %f" %(gof[0], gof[1], gof[2]) )
    m.GOF_plot(Entrada, Tempo_por_Passageiro, 1)
