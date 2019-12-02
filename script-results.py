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
embed_dim = []
lstm_out = []
dropout = []
batchsize = []
for arquivo in sys.argv[1:]:
    nomearquivo = arquivo.split("/")
    nomearquivo = nomearquivo[1][:-4]
    nomearquivo = re.split("[x|_|-]", nomearquivo)
    arq = open(arquivo, 'r')
    dados = arq.readline().replace("\n","").split(";")
    dados = list(map(float, dados))
    if not dados[0] in embed_dim:
        embed_dim.append(dados[0])
    if not dados[1] in lstm_out:
        lstm_out.append(dados[1])
    if not dados[2] in dropout:
        dropout.append(dados[2])
    if not dados[3] in batchsize:
        batchsize.append(dados[3])
    if not dados[0] in experimentos:
        experimentos[dados[0]] = {}
    if not dados[1] in experimentos[dados[0]]:
        experimentos[dados[0]][dados[1]] = {}
    if not dados[2] in experimentos[dados[0]][dados[1]]:
        experimentos[dados[0]][dados[1]][dados[2]] = {}
    if not dados[3] in experimentos[dados[0]][dados[1]][dados[2]]:
        experimentos[dados[0]][dados[1]][dados[2]][dados[3]] = {}
    exp = experimentos[dados[0]][dados[1]][dados[2]][dados[3]]
    exp["epochs"] = dados[4]
    exp["time"] = dados[5]
    exp["accuracy"] = dados[6]
    exp["f1"] = dados[7]
    exp["roc"] = dados[8]
    exp["confmat"] = [[dados[9],dados[10]],[dados[11],dados[12]]]

embed_dim.sort()
embed_dim = list(map(int, embed_dim))
lstm_out.sort()
lstm_out = list(map(int, lstm_out))
dropout.sort()
batchsize.sort()
batchsize = list(map(int, batchsize))

redes = []
for i in embed_dim:
    for j in lstm_out:
        redes.append([i,j])

pp.pprint(experimentos)

for embed in experimentos:
    for lstm in experimentos[embed]:
        for drop in experimentos[embed][lstm]:
            data = []
            for batch in batchsize:
                if batch in experimentos[embed][lstm][drop]:
                    data.append(experimentos[embed][lstm][drop][batch]["accuracy"])
            print(experimentos[embed][lstm][drop])
            print(data)
            print(batchsize)
            plt.plot(batchsize,data,'-',label="dropout " + str(drop))
        plt.xticks(batchsize)
        #for thread,color in zip(experimentos[instancia][entrada]['full'],colors):
        #    plt.plot(experimentos[instancia][entrada]['full'][thread]['iteracoes'], '-', label=str(thread)+" threads" , color=color)
        plt.xlabel('Acurácia')
        plt.ylabel('Tamanho do Batch')
        if nomearquivo[0] == 'lstm':
            plt.title("Acurácia para o LSTM\ncom " + str(embed) + " de embed_dim " + str(lstm) + " lstm_out")
        if nomearquivo[0] == 'cnn':
            plt.title("Acurácia para o CNN\ncom " + str(embed) + " de embed_dim " + str(lstm) + " filters")
        plt.legend(loc='upper left')
        #plt.savefig('graficos/'+entrada+'-'+instancia.replace(".","")+'-validinstance.svg', format="svg")
        #plt.show()
        plt.clf()

redesdados = {}
for i in batchsize:
    redesdados[i] = {}
barras = []
x = np.arange(len(redes))
width = 0.35
for embed,lstm in redes:
    for drop in experimentos[embed][lstm]:
        data = []
        for batch in batchsize:
            if not drop in redesdados[batch]:
                redesdados[batch][drop] = []
            if batch in experimentos[embed][lstm][drop]:
                redesdados[batch][drop].append(round(experimentos[embed][lstm][drop][batch]["f1"]*100,2))

acuraciavetor = []
f1vetor = []
tempovetor = []

for embed in embed_dim:
    if embed in experimentos:
        for lstm in lstm_out:
            if lstm in experimentos[embed]:
                for drop in dropout:
                    if drop in experimentos[embed][lstm]:
                        for batch in batchsize:
                            if batch in experimentos[embed][lstm][drop]:
                                acuraciavetor.append([experimentos[embed][lstm][drop][batch]["accuracy"],embed,lstm,drop,batch])
                                f1vetor.append([experimentos[embed][lstm][drop][batch]["f1"],embed,lstm,drop,batch,experimentos[embed][lstm][drop][batch]["epochs"],experimentos[embed][lstm][drop][batch]["time"],experimentos[embed][lstm][drop][batch]["accuracy"]])
                                tempovetor.append([experimentos[embed][lstm][drop][batch]["time"],embed,lstm,drop,batch])
acuraciavetor.sort()
f1vetor.sort()
tempovetor.sort()

positions = []
i=-width/2
while i <= width/2:
    positions.append(i)
    i=i+width/2

for drop in experimentos[embed_dim[0]][lstm_out[0]]:
    fig, ax = plt.subplots()
    for batch,i in zip(batchsize,range(len(batchsize))):
        ax.bar(x+positions[i], redesdados[batch][drop], width/2, label='Batch ' + str(batch))
    ax.set_ylabel('Pontuação F1 (%)')

    if nomearquivo[0] == 'lstm':
        ax.set_xlabel('Redes [embed_dim,lstm_out]')
    if nomearquivo[0] == 'cnn':
        ax.set_xlabel('Redes [embed_dim,filters]')
    ax.set_xticks(x)
    ax.set_xticklabels(redes, fontsize=8)
    ax.legend(ncol=3)
    #(loc='lower right')
    #plt.ylim(melhoracuracia-melhoracuracia*0.1,melhoracuracia+melhoracuracia*0.1)
    plt.ylim(0,80)
    #plt.title("Acurácia das diferentes redes LSTM\n com dropout " + str(drop) + " para os diferentes tamanhos de batch")

    if nomearquivo[0] == 'lstm':
        plt.savefig('graficos/lstm-f1-'+str(drop).replace('.','')+'-'+nomediretorio[1]+'.svg', format="svg")
    if nomearquivo[0] == 'cnn':
        plt.savefig('graficos/cnn-f1-'+str(drop).replace('.','')+'-'+nomediretorio[1]+'.svg', format="svg")
    fig.tight_layout()

    #plt.show()
    plt.clf()


embedcontrole = 0
embedantigo = 0
lstmcontrole = 0
lstmantigo = 0
dropoutcontrole = 0
dropoutantigo = dropout[0]
mediaf1 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1512 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf11024 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf12048 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf150 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1100 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1200 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1300 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1196 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaf1250 = {0.3 : [], 0.5 : [], 0.7 : []}
mediaepocas = {0.3 : [], 0.5 : [], 0.7 : []}
mediatempo = {0.3 : [], 0.5 : [], 0.7 : []}
for embed in embed_dim:
    if embed in experimentos:
        for lstm in lstm_out:
            if lstm in experimentos[embed]:
                for drop in dropout:
                    if drop in experimentos[embed][lstm]:
                        for batch in batchsize:
                            texto = ''
                            if batch in experimentos[embed][lstm][drop]:
                                if embed != embedantigo:
                                    if embedantigo != 0:
                                        texto = texto + '\Xhline{4\\arrayrulewidth}\n'
                                    embedantigo = embed
                                elif lstm != lstmantigo:
                                    if lstm == lstm_out[1]:
                                        texto = texto + '\Xhline{2\\arrayrulewidth}\n'
                                    lstmantigo = lstm
                                elif drop != dropoutantigo:
                                    if drop == dropout[1]:
                                        texto = texto + "\cline{2-7}\n"
                                    if drop == dropout[2]:
                                        texto = texto + "\cline{2-7}\n"
                                    dropoutantigo = drop
                                if lstmcontrole != lstm:
                                    lstmcontrole = lstm
                                    texto = texto + "\multirow{9}{*}{"
                                    if nomearquivo[0] == 'lstm':
                                        texto = texto + "LSTM ($"
                                    if nomearquivo[0] == 'cnn':
                                        texto = texto + "CNN ($"
                                    texto = texto + str(int(embed)) + "\\times"+str(int(lstm))+'$)} & '
                                else:
                                    texto = texto + "\t & "
                                    #texto = "LSTM ($"
                                    #texto = texto + str(int(embed)) + "\\times"+str(int(lstm))+'$) & $'
                                if dropoutcontrole != drop:
                                    dropoutcontrole = drop
                                    texto = texto + "\multirow{3}{*}{$" + str(drop).replace('.',',')+"$} &"
                                else:
                                    texto = texto + "\t & "
                                    #texto = texto + str(drop).replace('.',',')+"$ & $"
                                texto = texto + "$" + str(batch)+"$ & $"
                                texto = texto + str(int(experimentos[embed][lstm][drop][batch]["epochs"]))+"$ & $"
                                texto = texto + str(round(experimentos[embed][lstm][drop][batch]["time"],2)).replace('.',',')+"$ & $"
                                texto = texto + str(round(experimentos[embed][lstm][drop][batch]["accuracy"]*100,2)).replace('.',',')+"$ & $"
                                texto = texto + str(round(experimentos[embed][lstm][drop][batch]["f1"]*100,2)).replace('.',',')+"$\\\\"
                                print(texto)
                                mediaf1[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if batch == 512:
                                    mediaf1512[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if batch == 1024:
                                    mediaf11024[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if batch == 2048:
                                    mediaf12048[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if embed == 50:
                                    mediaf150[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if embed == 100:
                                    mediaf1100[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if embed == 200:
                                    mediaf1200[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if embed == 300:
                                    mediaf1300[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if lstm == 196:
                                    mediaf1196[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                if lstm == 250:
                                    mediaf1250[drop].append(experimentos[embed][lstm][drop][batch]["f1"])
                                mediaepocas[drop].append(int(experimentos[embed][lstm][drop][batch]["epochs"]))
                                mediatempo[drop].append(float(experimentos[embed][lstm][drop][batch]["time"]))
                                #print("LSTM ($"+str(int(embed))+"\\times"+str(int(lstm))+'$) & $'+str(drop).replace('.',',')+"$ & $"+str(batch)+"$ & $"+str(round(experimentos[embed][lstm][drop][batch]["accuracy"]*100,2)).replace('.',',')+"$ & $"+str(round(experimentos[embed][lstm][drop][batch]["f1"]*100,2)).replace('.',',')+"$\\\\")
print("MELHOR ACURACIA:")
print("\t" + str(acuraciavetor[len(acuraciavetor)-1]))
print("\t" + str(acuraciavetor[len(acuraciavetor)-1-1]))
print("\t" + str(acuraciavetor[len(acuraciavetor)-1-2]))
print("\t" + str(acuraciavetor[len(acuraciavetor)-1-3]))
print("\t" + str(acuraciavetor[len(acuraciavetor)-1-4]))
print("MELHOR F1:")
print("\t" + str(f1vetor[len(f1vetor)-1]))
print("\t" + str(f1vetor[len(f1vetor)-1-1]))
print("\t" + str(f1vetor[len(f1vetor)-1-2]))
print("\t" + str(f1vetor[len(f1vetor)-1-3]))
print("\t" + str(f1vetor[len(f1vetor)-1-4]))
print("MELHOR TEMPO:")
print("\t" + str(tempovetor[len(tempovetor)-1]))
print("\t" + str(tempovetor[len(tempovetor)-1-1]))
print("\t" + str(tempovetor[len(tempovetor)-1-2]))
print("\t" + str(tempovetor[len(tempovetor)-1-3]))
print("\t" + str(tempovetor[len(tempovetor)-1-4]))

if nomearquivo[0] == 'lstm':
    arquitetura = 'LSTM '
if nomearquivo[0] == 'cnn':
    arquitetura = 'CNN '
print(str(arquitetura)+"($"+str(f1vetor[len(f1vetor)-1][1])+"\\times"+str(f1vetor[len(f1vetor)-1][2])+"$) & $"+str(f1vetor[len(f1vetor)-1][3]).replace('.',',')+"$ & $"+str(f1vetor[len(f1vetor)-1][4])+"$ & $"+str(int(f1vetor[len(f1vetor)-1][5]))+"$ & $"+str(round(f1vetor[len(f1vetor)-1][6],2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-1][7]*100,2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-1][0]*100,2)).replace('.',',')+"$\\\\")
print(str(arquitetura)+"($"+str(f1vetor[len(f1vetor)-2][1])+"\\times"+str(f1vetor[len(f1vetor)-2][2])+"$) & $"+str(f1vetor[len(f1vetor)-2][3]).replace('.',',')+"$ & $"+str(f1vetor[len(f1vetor)-2][4])+"$ & $"+str(int(f1vetor[len(f1vetor)-2][5]))+"$ & $"+str(round(f1vetor[len(f1vetor)-2][6],2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-2][7]*100,2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-2][0]*100,2)).replace('.',',')+"$\\\\")
print(str(arquitetura)+"($"+str(f1vetor[len(f1vetor)-3][1])+"\\times"+str(f1vetor[len(f1vetor)-3][2])+"$) & $"+str(f1vetor[len(f1vetor)-3][3]).replace('.',',')+"$ & $"+str(f1vetor[len(f1vetor)-3][4])+"$ & $"+str(int(f1vetor[len(f1vetor)-3][5]))+"$ & $"+str(round(f1vetor[len(f1vetor)-3][6],2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-3][7]*100,2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-3][0]*100,2)).replace('.',',')+"$\\\\")
print(str(arquitetura)+"($"+str(f1vetor[len(f1vetor)-4][1])+"\\times"+str(f1vetor[len(f1vetor)-4][2])+"$) & $"+str(f1vetor[len(f1vetor)-4][3]).replace('.',',')+"$ & $"+str(f1vetor[len(f1vetor)-4][4])+"$ & $"+str(int(f1vetor[len(f1vetor)-4][5]))+"$ & $"+str(round(f1vetor[len(f1vetor)-4][6],2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-4][7]*100,2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-4][0]*100,2)).replace('.',',')+"$\\\\")
print(str(arquitetura)+"($"+str(f1vetor[len(f1vetor)-5][1])+"\\times"+str(f1vetor[len(f1vetor)-5][2])+"$) & $"+str(f1vetor[len(f1vetor)-5][3]).replace('.',',')+"$ & $"+str(f1vetor[len(f1vetor)-5][4])+"$ & $"+str(int(f1vetor[len(f1vetor)-5][5]))+"$ & $"+str(round(f1vetor[len(f1vetor)-5][6],2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-5][7]*100,2)).replace('.',',')+"$ & $"+str(round(f1vetor[len(f1vetor)-5][0]*100,2)).replace('.',',')+"$\\\\")


print("MEDIA F1:")
print("\t" + str(np.mean(mediaf1[0.3])))
print("\t" + str(np.mean(mediaf1[0.5])))
print("\t" + str(np.mean(mediaf1[0.7])))
#print("MEDIANA F1:")
#print("\t" + str(np.median(mediaf1[0.3])))
#print("\t" + str(np.median(mediaf1[0.7])))
print("\t" + str((mediaf1[0.3])))
print("\t" + str((mediaf1[0.5])))
print("\t" + str((mediaf1[0.7])))
print("MEDIA F1 batch 512:")
print("\t" + str(np.mean(mediaf1512[0.3])))
print("\t" + str(np.mean(mediaf1512[0.5])))
print("\t" + str(np.mean(mediaf1512[0.7])))
print("\t" + str((mediaf1512[0.3])))
print("\t" + str((mediaf1512[0.5])))
print("\t" + str((mediaf1512[0.7])))
print("MEDIA F1 batch 1024:")
print("\t" + str(np.mean(mediaf11024[0.3])))
print("\t" + str(np.mean(mediaf11024[0.5])))
print("\t" + str(np.mean(mediaf11024[0.7])))
print("\t" + str((mediaf11024[0.3])))
print("\t" + str((mediaf11024[0.5])))
print("\t" + str((mediaf11024[0.7])))
print("MEDIA F1 batch 2048:")
print("\t" + str(np.mean(mediaf12048[0.3])))
print("\t" + str(np.mean(mediaf12048[0.5])))
print("\t" + str(np.mean(mediaf12048[0.7])))
print("\t" + str((mediaf12048[0.3])))
print("\t" + str((mediaf12048[0.5])))
print("\t" + str((mediaf12048[0.7])))
print("MEDIA F1 embed 50:")
print("\t" + str(np.mean(mediaf150[0.3])))
print("\t" + str(np.mean(mediaf150[0.5])))
print("\t" + str(np.mean(mediaf150[0.7])))
print("\t" + str((mediaf150[0.3])))
print("\t" + str((mediaf150[0.5])))
print("\t" + str((mediaf150[0.7])))
print("MEDIA F1 embed 100:")
print("\t" + str(np.mean(mediaf1100[0.3])))
print("\t" + str(np.mean(mediaf1100[0.5])))
print("\t" + str(np.mean(mediaf1100[0.7])))
print("\t" + str((mediaf1100[0.3])))
print("\t" + str((mediaf1100[0.5])))
print("\t" + str((mediaf1100[0.7])))
print("MEDIA F1 embed 200:")
print("\t" + str(np.mean(mediaf1200[0.3])))
print("\t" + str(np.mean(mediaf1200[0.5])))
print("\t" + str(np.mean(mediaf1200[0.7])))
print("\t" + str((mediaf1200[0.3])))
print("\t" + str((mediaf1200[0.5])))
print("\t" + str((mediaf1200[0.7])))
print("MEDIA F1 embed 300:")
print("\t" + str(np.mean(mediaf1300[0.3])))
print("\t" + str(np.mean(mediaf1300[0.5])))
print("\t" + str(np.mean(mediaf1300[0.7])))
print("\t" + str((mediaf1300[0.3])))
print("\t" + str((mediaf1300[0.5])))
print("\t" + str((mediaf1300[0.7])))
print("MEDIA F1 lstm 196:")
print("\t" + str(np.mean(mediaf1196[0.3])))
print("\t" + str(np.mean(mediaf1196[0.7])))
print("\t" + str(np.mean(mediaf1196[0.5])))
print("\t" + str((mediaf1196[0.3])))
print("\t" + str((mediaf1196[0.5])))
print("\t" + str((mediaf1196[0.7])))
print("MEDIA F1 lstm 250:")
print("\t" + str(np.mean(mediaf1250[0.3])))
print("\t" + str(np.mean(mediaf1250[0.5])))
print("\t" + str(np.mean(mediaf1250[0.7])))
print("\t" + str((mediaf1250[0.3])))
print("\t" + str((mediaf1250[0.5])))
print("\t" + str((mediaf1250[0.7])))
print("MEDIA TEMPO:")
print("\t" + str(np.mean(mediatempo[0.3])))
print("\t" + str(np.mean(mediatempo[0.5])))
print("\t" + str(np.mean(mediatempo[0.7])))
print("\t" + str(np.mean(mediatempo[0.3]+mediatempo[0.5]+mediatempo[0.7])))
print("MEDIA EPOCAS:")
print("\t" + str(np.mean(mediaepocas[0.3])))
print("\t" + str(np.mean(mediaepocas[0.5])))
print("\t" + str(np.mean(mediaepocas[0.7])))
print("\t" + str(np.mean(mediaepocas[0.3]+mediaepocas[0.5]+mediaepocas[0.7])))
