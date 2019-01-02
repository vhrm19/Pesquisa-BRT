import numpy as np
import csv

dataset = []
data = csv.reader(open("csv desembarque.csv","r"), delimiter=';')
for line in data:
    line = [float(elemento) for elemento in line]
    dataset.append(line)
Tempo_por_Passageiro, Passageiros, Em_pe = [], [], []
for i in range(len(dataset)):
    Tempo_por_Passageiro.append(dataset[i][0])
    Passageiros.append(dataset[i][1])
    Em_pe.append(dataset[i][2])

Tempo_por_Passageiro = np.array(Tempo_por_Passageiro)
Passageiros = np.array(Passageiros)
Em_pe = np.array(Em_pe)

for i in range(len(Tempo_por_Passageiro)):
    Tempo_por_Passageiro[i] = (Tempo_por_Passageiro[i] - 2.0253333) / 1.397990069
    Passageiros[i] = (Passageiros[i] - 2.66) / 1.975461713
    Em_pe[i] = (Em_pe[i] - 1.6) / 0.832993128
for i in range(len(dataset)):
    dataset[i][0] = str((float(Tempo_por_Passageiro[i])))
    dataset[i][1] = str((float(Passageiros[i])))
    dataset[i][2] = str((float(Em_pe[i])))

arq = open('csv desembarque padronizado.csv', 'w')

for i in range(len(dataset)):
    arq.write(dataset[i][0])
    arq.write(';')
    arq.write(dataset[i][1])
    arq.write(';')
    arq.write(dataset[i][2])
    if i < len(dataset)-1:
        arq.write('\n')
arq.close()