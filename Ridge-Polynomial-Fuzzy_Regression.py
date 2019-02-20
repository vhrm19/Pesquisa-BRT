import numpy as np
from gurobipy import *
import csv
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV

class Ridge_poly_fuzzy:

    def fit(self, x, y, h):

        ac = RidgeCV(alphas=np.arange(1e-3, 1, 0.009), normalize=True, cv = len(x)).fit(x, y)

        model = Model()
        model.setParam("OutputFlag", 0)

        aL = {}
        aR = {}

        for i in range(len(x.T) + 3):
            aL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
            aR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

        obj = quicksum( (aL[0] + aL[1] * x[i] + aL[2] * x[i]**2 + aL[3] * x[i]**3) +
                        (aR[0] + aR[1] * x[i] + aR[2] * x[i]**2 + aR[3] * x[i]**3) for i in range(len(x)))

        model.setObjective(obj, GRB.MINIMIZE)

        for i in range(len(x)):
            model.addConstr(h* (aL[0] + aL[1] * x[i] + aL[2] * x[i]**2 + aL[3] * x[i]**3) - (ac.coef_ * x[i] + ac.intercept_) >= -y[i])
            model.addConstr(h* (aR[0] + aR[1] * x[i] + aR[2] * x[i]**2 + aR[3] * x[i]**3) + (ac.coef_ * x[i] + ac.intercept_) >= y[i])
            model.addConstr((aL[0] + aL[1] * x[i] + aL[2] * x[i]**2 + aL[3] * x[i]**3) >= 0)
            model.addConstr((aR[0] + aR[1] * x[i] + aR[2] * x[i]**2 + aR[3] * x[i]**3) >= 0)

        model.optimize()

        Lh = []
        Rh = []
        L = []
        R = []
        C = []

        for i in range(len(x)):  
            Lh.append(float((ac.coef_ * i + ac.intercept_) - h* (aL[0].x + aL[1].x * i + aL[2].x * i**2 + aL[3].x * i**3)))
            Rh.append(float((ac.coef_ * i + ac.intercept_) + h* (aR[0].x + aR[1].x * i + aR[2].x * i**2 + aR[3].x * i**3)))
            L.append(float((ac.coef_ * i + ac.intercept_) - (aL[0].x + aL[1].x * i + aL[2].x * i**2 + aL[3].x * i**3)))
            R.append(float((ac.coef_ * i + ac.intercept_) + (aR[0].x + aR[1].x * i + aR[2].x * i**2 + aR[3].x * i**3)))
            C.append(float(ac.coef_ * i + ac.intercept_))

        plt.plot(x, y, "bo")
        plt.plot(range(len(x)), R, label = 'C + R')
        plt.plot(range(len(x)), Rh, label = 'C + hR')
        plt.plot(range(len(x)), C, label = 'C')
        plt.plot(range(len(x)), Lh, label = 'C + hL')
        plt.plot(range(len(x)), L, label = 'C + L')
        plt.legend()
        plt.show()

        ymid = []

        for i in range(len(x)):
            ymid.append((Lh[i] + Rh[i])/2)
        SST = 0
        for i in range(len(x)):
            SST += (y[i] - Lh[i]) * (y[i] - Lh[i]) + \
                (Rh[i] - y[i]) * (Rh[i] - y[i])
        SSR = 0
        for i in range(len(x)):
            SSR += (ymid[i] - Lh[i]) * (ymid[i] - Lh[i]) + \
                (Rh[i] - ymid[i]) * (Rh[i] - ymid[i])

        print("IC:", float(SSR / SST))

        print("AFS:", float(sum(h * abs(Rh[i] - Lh[i]) for i in range(len(x)))))

        print("MFC:", float(sum( C[i] / abs(Rh[i] - Lh[i]) for i in range(len(x)))))
                                            
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
    dataset = Passageiros(data)[0] # Passageiros: 0 = Poucos, 1 = Medio, 2 = Muitos
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    m = Ridge_poly_fuzzy()
    m.fit(Entrada, Tempo_por_Passageiro, 0.6)
