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




      
trainDataset = spark.read.option("header", True).option("delimiter", "|").csv("../data/labeled_suspicious_text_balanced.csv")
trainDataset.show()


document = DocumentAssembler().setInputCol("tweet").setOutputCol("document")

sentences = SentenceDetector().setInputCols(["document"]).setOutputCol("sentences")
token = Tokenizer().setInputCols("sentences").setOutputCol("token")
normal = Normalizer().setInputCols("token").setOutputCol("normal")
cleaner = StopWordsCleaner().setInputCols("normal").setOutputCol("stop")
lemma = LemmatizerModel.pretrained('lemma_antbnc').setInputCols(["stop"]).setOutputCol("lemma")
use = UniversalSentenceEncoder.pretrained().setInputCols(["document","lemma"]).setOutputCol("sentence_embeddings")


classsifierdl = ClassifierDLApproach()\
.setInputCols(["sentence_embeddings"])\
.setOutputCol("class").setLabelColumn("class")\
.setMaxEpochs(20)\
.setEnableOutputLogs(True)

use_clf_pipeline = Pipeline(
    stages = [
        document,
        sentences,
        token,
        normal,
        cleaner,
        lemma,
        use,
        classsifierdl
    ])

use_pipelineModel = use_clf_pipeline.fit(trainDataset)
use_pipelineModel.save("../Models/spark_model")



'''
light_model = LightPipeline(use_pipelineModel)
while(True):
  text = input("in:\n")
  print(light_model.annotate(text)['class'][0])
'''


