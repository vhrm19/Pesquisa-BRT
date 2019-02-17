import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV

class Regressions:
    def __init__(self, model):
        self.model = model
        if model == 0:
            self.lr = RidgeCV
            self.name = 'Ridge'
        elif model == 1:
            self.lr = LassoCV
            self.name = 'Lasso'
        
    def fit(self, Input, Output):
        self.Input = Input
        self.Output = Output
        lm = self.lr(alphas=np.arange(1e-3, 1, 0.009), normalize=False, cv = len(self.Input)).fit(Input, Output)

        r2 = 'LOOCV R2:%f' %lm.score(self.Input, self.Output)

        plt.plot(self.Input, self.Output, 'bo')
        plt.xlabel('Entrada')
        plt.ylabel('Tempo por Passageiro')
        x = np.arange(1,15)
        plt.plot(x, eval(str("%f + %f*x" %(lm.intercept_, lm.coef_))), label = (self.name, r2))

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

    for i in range(2):
        model = Regressions(model = i)
        model.fit(Entrada, np.ravel(Tempo_por_Passageiro))

    plt.legend()
    plt.show()
