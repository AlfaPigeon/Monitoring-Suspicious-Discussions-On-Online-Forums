# conda activate spacynlp
# python3.8

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

import os
import numpy as np
import pandas as pd

import spacy
import spacy_universal_sentence_encoder

nlp = spacy.load('en_core_web_lg')

def predict(model,sentence):
    return model.predict(np.array([nlp(sentence).vector]))[0][0]

model = tf.keras.models.load_model('../Models/tf_spacy_oguz')


pagename=[]
author=[]
title=[]
comment=[]
suspicious=[]


df = pd.read_csv("../data/reddit.csv")
for i in range(len(df)):
    guess=2
    try:
        guess=predict(model,df["comment"][i])
    except:
        continue
    
    if(guess > 0.5):
        guess = 1
    else:
        guess = 0
    pagename.append(df["pagename"][i])
    author.append(df["author"][i])
    title.append(df["title"][i])
    comment.append(df["comment"][i])
    suspicious.append(guess)




df_2 = pd.DataFrame(data= { 'pagename': pagename ,'author': author ,'title': title ,'comment': comment ,'suspicious': suspicious })
df_2 = df_2.replace('\n',' ', regex=True)
df_2 = df_2.replace('"',' ', regex=True)
df_2.to_csv('../data/classified_spacy_reddit.csv', index=False,sep="|") 








