import numpy as np
import csv
from sklearn.svm import SVR

dataset = np.array(list(csv.reader(open("csv desembarque.csv","r"), delimiter=';'))).astype("float")
Entrada = np.hsplit(dataset, (0,1))[2]
Tempo_por_Passageiro = np.hsplit(dataset, (0,1))[1]
Tempo_por_Passageiro = np.reshape(Tempo_por_Passageiro, len(Tempo_por_Passageiro))

svr = SVR(C=3, epsilon=0.01, kernel='rbf', gamma = 'scale')
mse = 0
for i in range(len(Tempo_por_Passageiro)):
    svr.fit(np.delete(Entrada, i, 0), np.delete(Tempo_por_Passageiro, i))
    mse += (Tempo_por_Passageiro[i] - svr.predict([Entrada[i]]))**2

print("LOOCV:", float(mse / len(Tempo_por_Passageiro)))
