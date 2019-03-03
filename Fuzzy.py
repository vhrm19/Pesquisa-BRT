import numpy as np
from gurobipy import *
import csv
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10}

matplotlib.rc('font', **font)
matplotlib.style.use('ggplot')

def plota(x, y, ac, awL, awR, size):#, xname, yname, size):
    plt.plot(x, y, 'o', markersize=2, label='Dados')
    xvalues = np.arange(min(x), max(x), 0.1)
    ylow = (ac - awL.x) * xvalues
    ymid = ac * xvalues
    yhigh = (ac + awR.x) * xvalues
    superior = plt.plot(xvalues, ylow, 'b--', label='Limite Inferior')
    centroide = plt.plot(xvalues, ymid, 'k--', label='Centroide')
    inferior = plt.plot(xvalues, yhigh, 'r--', label='Limite Superior')
    #plt.legend()
    #plt.xlabel(xname[0], fontsize=12)
    #plt.ylabel(yname, fontsize=12)
    #plt.savefig('imgs/fuzzy' + yname, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.cla()


def otimiza(y, x, h, plotaIC='false'):

    n = len(y)

    model = Model("fuzzy")
    model.setParam("OutputFlag", 0)
    awL = {}
    awR = {}
    size = 1

#    h = 0

    awL = model.addVar(lb=-0.0, name="awL")
    awR = model.addVar(lb=-0.0, name="awR")

    #num, c = np.linalg.lstsq(x, y, rcond=None)[0]
    #print(m)
    ac, resid = np.linalg.lstsq(x, y, rcond=-1)[:2]
    ac = ac[0]

    x = x.flatten()
    y = y.flatten()

    model.setObjective(quicksum((float(np.dot(x, x.transpose()))
                                 * (awL + awR) * (awL + awR))
                                for j in range(size)),
                       GRB.MINIMIZE)

    for i in range(n):
        model.addConstr(
            quicksum( - (ac * x[i]) + (1 - h) * abs(x[i]) * awL) >= -y[i])

    for i in range(n):
        model.addConstr(
            quicksum( (ac * x[i]) + (1 - h) * abs(x[i]) * awR) >= y[i])

    model.optimize()

#    print(yname)
#    print(xname)
#    print(ic)

    if plotaIC == 'false':
        print(awL)
        print(awR)
        plota(x, y, ac, awL, awR, 1)
        return [ac, (ac - awL.x), (ac + awR.x)]
    if plotaIC == 'true':
        return model, ac, awL, awR

def u_membership(y, ymid, yhigh, ylow):
    if ymid == y:
        return 1
    elif (y > ylow) and (y < ymid):
        return (y - ylow)/(ymid-ylow)
    elif (y < yhigh) and (y > ymid):
        return (yhigh - y)/(yhigh-ymid)
    else:
        return 0

def IC(x, y, ac, awL, awR, h, size, verbose = True):
    ylow = []
    yhigh = []
    ymid = []

    for i in range(len(y)):
        ylow.append((ac - awL.x) * x[i])

    for i in range(len(y)):
        yhigh.append((ac + awR.x) * x[i])

    for i in range(len(y)):
        ymid.append(ac[0] * x[i])

    SST = 0
    for i in range(len(y)):
        SST += (y[i] - ylow[i]) * (y[i] - ylow[i]) + \
            (yhigh[i] - y[i]) * (yhigh[i] - y[i])

    SSR = 0
    for i in range(len(y)):
        SSR += (ymid[i] - ylow[i]) * (ymid[i] - ylow[i]) + \
            (yhigh[i] - ymid[i]) * (yhigh[i] - ymid[i])

    IC = SSR / SST

    u = []
    for i in range(len(y)):
        u.append(u_membership(y[i], ymid[i], yhigh[i], ylow[i]))

    AFS = float(sum(h * abs(yhigh[i] - ylow[i]) for i in range(len(x))))
    MFC = float(sum( u[i] / abs(yhigh[i] - ylow[i]) for i in range(len(x))))

    if verbose == True:
        # higher is better
        print("IC:", float(SSR / SST))
        # lower is better
        print("AFS:", AFS)
        # higher is better
        print("MFC:", MFC)

    return IC, MFC, AFS


def plotaIC(y, x, size):
    h = 0
    IClist = []
    hlist = []
    awRlist = []
    awLlist = []

    #nomeia = y.name

    for i in range(0, 19):
        h += 0.05
        print(h)
        hlist.append(h)
        model, ac, awL, awR = otimiza(y, x, h, plotaIC='true')
        awRlist.append(awR.x)
        awLlist.append(awL.x)
        IClist.append(IC(x, y, ac, awL, awR, h, size, verbose = False))

    x = hlist
    y = IClist

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('índice h')
    ax.set_ylabel('IC')
    ax.set_zlabel('aR')
    ax.set_facecolor('white')
    ax.plot(x, y, awRlist)
    plt.savefig('imgs/IC_awR_', bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.cla()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('índice h')
    ax.set_ylabel('IC')
    ax.set_zlabel('aL')
    ax.set_facecolor('white')
    ax.plot(x, y, awLlist)
    plt.savefig('imgs/IC_awL_', bbox_inches='tight')
    plt.clf()
    plt.cla()

def otimiza_h( x, y):
    model, ac, awL, awR = otimiza(y, x, 0, plotaIC='true')
    IC(x, y, ac, awL, awR, 0, 1, verbose = True)

    ylow = []
    yhigh = []
    ymid = []

    for i in range(len(y)):
        ylow.append((ac - awL.x) * x[i])

    for i in range(len(y)):
        yhigh.append((ac + awR.x) * x[i])

    for i in range(len(y)):
        ymid.append(ac[0] * x[i])

    u = []
    for i in range(len(y)):
        u.append(u_membership(y[i], ymid[i], yhigh[i], ylow[i]))

    z0 = sum( u[i] / abs(yhigh[i] - ylow[i]) for i in range(len(x)))
    p0 = sum( (1-u[i]) / abs(yhigh[i] - ylow[i]) for i in range(len(x)))

    if z0 <= p0:
        h = 0.5 * (1 - z0/p0)
    else:
        h = 0

    print("h otimo:", h[0])

    model, ac, awL, awR = otimiza(y, x, h, plotaIC='true')
    IC(x, y, ac, awL, awR, h, 1, verbose = True)

def varGOF(x, y):
    MFClist = []
    AFSlist = []
    IClist = []

    hlist = np.arange(0, 1, 0.05)

    for h in hlist:
        model, ac, awL, awR = otimiza(y, x, h, plotaIC='true')
        ic, MFC, AFS = IC(x, y, ac, awL, awR, h, 1, verbose = False)
        MFClist.append(MFC)
        IClist.append(ic)
        AFSlist.append(AFS)
    
    plt.plot(hlist, IClist, label = 'IC')
    plt.plot(hlist, AFSlist, label = 'AFS')
    plt.plot(hlist, MFClist, label = 'MFC')
    plt.xlabel('h')
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
    dataset = Passageiros(data)[1] # Passageiros: 0 = Poucos, 1 = Medio, 2 = Muitos
    Entrada = np.hsplit(dataset, (0,1))[2]
    Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]

    otimiza_h(Entrada, Tempo_por_Passageiro)
    varGOF(Entrada, Tempo_por_Passageiro)

    #plotaIC(Entrada, Tempo_por_Passageiro, 1)
