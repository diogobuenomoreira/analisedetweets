#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import pprint
import numpy as np
import scipy.stats
import os
import re

if len(sys.argv) == 1:
    print("ERRO: não há arquivo para leitura")
    print("\tpython script-graficos.py <nome arquivo>")
    exit(1)

nomediretorio = sys.argv[1].split("/")
nomediretorio = nomediretorio[0].split('-')

print(sys.argv)

graficosdir = 'graficos'
if not os.path.exists(graficosdir):
    os.makedirs("./" + graficosdir)

colors = ('#e57914', '#18ce64', '#1ca8ef', '#ce18a9', '#d13714', '#aed114', '#f9bb7a', '#a55e13',
            '#e3c712', '#85e312', '#3a21cc', '#cc23c4', '#36d9d1', '#a437de')
pp = pprint.PrettyPrinter(indent=2)
experimentos = {}
tempos = []
for arquivo in sys.argv[1:]:
    nomearquivo = arquivo.split("/")
    nomearquivo2 = nomearquivo[0].split('-')
    nomearquivo = nomearquivo[1][:-4]
    nomearquivo = re.split("[x|_|-]", nomearquivo)
    arq = open(arquivo, 'r')
    dados = arq.readline().replace("\n","").split(";")
    dados = list(map(float, dados))
    if not dados[0] in experimentos:
        experimentos[dados[0]] = {}
    if not dados[1] in experimentos[dados[0]]:
        experimentos[dados[0]][dados[1]] = {}
    if not dados[2] in experimentos[dados[0]][dados[1]]:
        experimentos[dados[0]][dados[1]][dados[2]] = {}
    if not dados[3] in experimentos[dados[0]][dados[1]][dados[2]]:
        experimentos[dados[0]][dados[1]][dados[2]][dados[3]] = {}
    if not dados[4] in experimentos[dados[0]][dados[1]][dados[2]][dados[3]]:
        experimentos[dados[0]][dados[1]][dados[2]][dados[3]][dados[4]] = {}
    if not dados[5] in experimentos[dados[0]][dados[1]][dados[2]][dados[3]][dados[4]]:
        experimentos[dados[0]][dados[1]][dados[2]][dados[3]][dados[4]][dados[5]] = {}
    exp = experimentos[dados[0]][dados[1]][dados[2]][dados[3]][dados[4]][dados[5]]
    exp["epochs"] = dados[6]
    exp["time"] = dados[7]
    tempos.append(dados[7])
    exp["accuracy"] = dados[8]
    exp["f1"] = dados[9]
    exp["roc"] = dados[10]
    exp["confmat"] = [[dados[11],dados[12]],[dados[13],dados[14]]]



pp.pprint(experimentos)
trains = [0, 1]
vocabs = [1000, 2000, 2500]
if nomearquivo[0] == 'cnn':
    kaggleexp = [[100, 250, 0.3, 512],
                [100, 196, 0.7, 512],
                [300, 250, 0.3, 512],
                [200, 196, 0.3, 512],
                [200, 196, 0.5, 512]]
    gitexp = [[50, 196, 0.7, 512],
          [100, 196, 0.7, 512],
          [100, 196, 0.5, 2048],
          [200, 250, 0.7, 512],
          [300, 250, 0.5, 1024]]
if nomearquivo[0] == 'lstm':
    kaggleexp = [[50, 250, 0.7, 512],
              [100, 196, 0.3, 1024],
              [100, 196, 0.5, 512],
              [300, 250, 0.5, 2048],
              [50, 250, 0.5, 1024]]
    gitexp = [[100, 250, 0.7, 512],
            [300, 250, 0.7, 512],
            [200, 250, 0.3, 512],
            [200, 196, 0.7, 1024],
            [200, 196, 0.7, 512]]
experimentosfinal = []
maior = 0.0
maiorvetor = []
if nomearquivo2[2] == "kaggle":
    for k in kaggleexp:
        exp = []
        exp = k
        for i in vocabs:
            for j in trains:
                exp.append(i)
                exp.append(j)
                exp.append(experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1'])
                if maior < experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1']:
                    maior = experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1']
                    maiorvetor = [k[0],k[1],k[2],k[3],i,j,maior]
        experimentosfinal.append(exp)

if nomearquivo2[2] == "git":
    for k in gitexp:
        exp = []
        exp = k
        for i in vocabs:
            for j in trains:
                exp.append(i)
                exp.append(j)
                exp.append(experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1'])
                if maior < experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1']:
                    maior = experimentos[k[0]][k[1]][k[2]][k[3]][i][j]['f1']
                    maiorvetor = [k[0],k[1],k[2],k[3],i,j,maior]
        experimentosfinal.append(exp)
print("MAIOR F1:")
print(maiorvetor)
print("MEDIA TEMPO:")
print(np.mean(tempos))
if nomearquivo[0] == 'lstm':
    arquitetura = 'LSTM '
if nomearquivo[0] == 'cnn':
    arquitetura = '\multicolumn{1}{ l|| }{CNN '
for i in experimentosfinal:
    texto = arquitetura+"($"
    texto = texto + str(i[0]) + "\\times"
    texto = texto + str(i[1]) + "$) $"
    texto = texto + str(i[2]).replace('.',',') + "$ $"
    texto = texto + str(i[3]) + "$} & $"
    texto = texto + str(round(i[6]*100,2)).replace('.',',') + "$ & $"
    texto = texto + str(round(i[9]*100,2)).replace('.',',') + "$ & $"
    texto = texto + str(round(i[12]*100,2)).replace('.',',') + "$ & $"
    texto = texto + str(round(i[15]*100,2)).replace('.',',') + "$ & $"
    texto = texto + str(round(i[18]*100,2)).replace('.',',') + "$ & $"
    texto = texto + str(round(i[21]*100,2)).replace('.',',') + "$\\\\"
    print(texto)
