# conda activate sparknlp
# python

import os
java8_location= '/usr/lib/jvm/java-8-openjdk-amd64'
os.environ['JAVA_HOME'] = java8_location
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *              
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

import pandas as pd

spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.6.4")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()




model = PipelineModel.load("../Models/spark_model")
light_model = LightPipeline(model)

pagename=[]
author=[]
title=[]
comment=[]
suspicious=[]


df = pd.read_csv("../data/reddit.csv")
for i in range(len(df)):
    guess=2
    try:
        guess=light_model.annotate(df["comment"][i])['class'][0]
    except:
        continue
    
    if(str(guess) != "2"):
        guess = 1
    else:
        guess = 0
    pagename.append(df["pagename"][i])
    author.append(df["author"][i])
    title.append(df["title"][i])
    comment.append(df["comment"][i])
    suspicious.append(guess)




df_2 = pd.DataFrame(data= { 'pagename': pagename ,'author': author ,'title': title ,'comment': comment ,'suspicious': suspicious })

df_2.to_csv('../data/classified_spark_reddit.csv', index=False,sep="|") 
