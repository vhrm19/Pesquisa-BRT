import numpy as np
from gurobipy import *
import csv

dataset = []
data = csv.reader(open("csv desembarque.csv","r"), delimiter=';')
for line in data:
    line = [float(elemento) for elemento in line]
    dataset.append(line)

Tempo_por_Passageiro=[]
for i in range(len(dataset)):
    Tempo_por_Passageiro.append(dataset[i][0])

Passageiros=[]
for i in range(len(dataset)):
    Passageiros.append(dataset[i][1])

Em_pe=[]
for i in range(len(dataset)):
    Em_pe.append(dataset[i][2])

x = np.array(Passageiros)
y = np.array(Em_pe)
