#Para cofidicacao de caracteres
# -*- coding: utf-8 -*-
import sys
from importlib import reload
import pandas as pd
reload(sys)
#sys.setdefaultencoding('utf8')

#Bibliotecas
import tweepy
from textblob import TextBlob
from textblob import exceptions
import numpy as np
from tweepy import TweepError

#Chaves de autorizacao
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

#Abrir conexao
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Entrada
#arquivo = raw_input("Arquivo do Artigo: ");
arquivo = "data/NAACL_SRW_2016.csv"

#Obter dados
#tweet = api.get_status(id_of_tweet)
#print(tweet.)

with open(arquivo) as f:
    content = f.readlines()

content = [x.strip() for x in content]


id_out = 9
labels =[]
tweets = []
controle = 0
for c in content:
    id_of_tweet = c.split(',')
    controle = controle + 1
    if controle < 1937:
        continue
    try:
        tweet = api.get_status(id_of_tweet[0])
        if(id_of_tweet[1] == "racism" or id_of_tweet[1] == "sexism"):
            label = 1
        else:
            label = 0
        labels.append(label)
        tweets.append(tweet.text.replace('\n', ' '))
        #fd.write(str(id_out)+","+str(label)+","+tweet.text+"\n")
        print(str(id_out))
        id_out+=1
    except:
        print("Tweet not found")
dict = {'label': labels, 'tweet': tweets}
df = pd.DataFrame(dict)

df.to_csv('data/train3.csv')
