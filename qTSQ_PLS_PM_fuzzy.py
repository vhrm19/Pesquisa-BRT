import numpy as np
from gurobipy import *
import csv
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class qTSQ_PLS_PM_fuzzy:

    def plota(self, x, y, ac, awL, awR, xname, yname, size):
        plt.plot(x, y, 'o', markersize=2, label='Dados')
        xvalues = np.arange(np.amin(x), np.amax(x), 0.1)
        ylow = -awL[0].x + [(ac[i] - awL[i + 1].x) *
                            xvalues for i in range(size)][0]
        ymid = [ac[i] * xvalues for i in range(size)][0]
        yhigh = awR[0].x + [(ac[i] + awR[i + 1].x) *
                            xvalues for i in range(size)][0]
        superior = plt.plot(xvalues, ylow, 'b--', label='Limite Inferior')
        centroide = plt.plot(xvalues, ymid, 'k--', label='Centroide')
        inferior = plt.plot(xvalues, yhigh, 'r--', label='Limite Superior')
        plt.legend()
        plt.xlabel(xname[0], fontsize=12)
        plt.ylabel(yname, fontsize=12)
        #plt.savefig('imgs/fuzzy' + yname, bbox_inches='tight')
        #plt.clf()
        #plt.cla()
        plt.show()


    def otimiza(self, y, x, size, h, method='fuzzy', plotaIC='false'):

        n = len(y)

        model = Model("qTSQ-PLS-PM with fuzzy scheme")
        model.setParam("OutputFlag", 0)
        awL = {}
        awR = {}

    #    h = 0

        for i in range(size + 1):
            awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
            awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)
    
        ac, resid = np.linalg.lstsq(x, y, rcond = -1)[:2]

        #yname = y.name
        #xname = x.columns.values
    #    print(['y: ' + yname])
    #    print('x: ' + xname)

        #y = y.values
        #x = x.values

        model.setObjective(quicksum((float(np.dot(x[:, j], x[:, j].transpose()))
                                    * (awL[j + 1] + awR[j + 1]) * (awL[j + 1] + awR[j + 1]))
                                    for j in range(size))
                        + (awL[0] + awR[0]) * (awL[0] + awR[0]),
                        GRB.MINIMIZE)

        # Lembrar que no for não vai N

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
    #    print(awL)
    #    print(awR)
    #    plota(x, y, ac, awL, awR, xname, yname, size)
    #    ic = IC(x, y, ac, awL, awR, size)

    #    print(yname)
    #    print(xname)
    #    print(ic)

        if plotaIC == 'false':
            return [ac[i] for i in range(size)], [(ac[i] - awL[i + 1].x) for i in range(size)], [(ac[i] + awR[i + 1].x) for i in range(size)]
        if plotaIC == 'true':
            return model, ac, awL, awR


    def IC(self, x, y, ac, awL, awR, size):
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

        IC = SSR / SST
        return IC


    def plotaIC(self, y, x, size):
        h = 0
        IClist = []
        hlist = []
        awRlist = []
        awLlist = []

        #nomeia = y.name

        for i in range(0, 19):
            h += 0.05
            hlist.append(h)
            model, ac, awL, awR = self.otimiza(y, x, size, h, plotaIC='true')
            awRlist.append(awR[1].x)
            awLlist.append(awL[1].x)
            IClist.append(self.IC(x, y, ac, awL, awR, size))

        x = hlist
        y = IClist

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('índice h')
        ax.set_ylabel('IC')
        ax.set_zlabel('awR')
        #ax.set_axis_bgcolor('white')
        ax.plot(x, y, awRlist, label = 'awR')
        #plt.savefig('imgs/IC_awR_' + nomeia, bbox_inches='tight')
        #plt.clf()
        #plt.cla()

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #ax.set_xlabel('índice h')
        #ax.set_ylabel('IC')
        #ax.set_zlabel('awL')
        #ax.set_axis_bgcolor('white')
        ax.plot(x, y, awLlist, label = 'awL')
        #plt.savefig('imgs/IC_awL_' + nomeia, bbox_inches='tight')
        #plt.clf()
        #plt.cla()
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

    data = np.array(list(csv.reader(open("csv embarque.csv","r"), delimiter=';'))).astype("float")
    
    dataset = Passageiros(data)[1] # Passageiros: 0 = Poucos, 1 = Medio, 2 = Muitos

    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    m = qTSQ_PLS_PM_fuzzy()

    model, ac, awL, awR = m.otimiza(Tempo_por_Passageiro, Entrada, 1, 0.5, method='fuzzy', plotaIC='true')
    print('IC =', m.IC(Entrada, Tempo_por_Passageiro, ac, awL, awR, 1))
    m.plota(Entrada, Tempo_por_Passageiro, ac, awL, awR, "Entrada", "T/P", 1)
    m.plotaIC(Tempo_por_Passageiro, Entrada, 1)
