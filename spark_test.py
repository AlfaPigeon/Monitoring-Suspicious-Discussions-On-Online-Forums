import os
java8_location= '/usr/lib/jvm/java-8-openjdk-amd64' # Set your own
os.environ['JAVA_HOME'] = java8_location

from sparknlp.base import *
from sparknlp.annotator import *              
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
import pandas as pd


spark = sparknlp.start()
sentences = [
  ['Hello, this is an example sentence'],
  ['And this is a second sentence.'],
  ['I\'m swimming']
]

# spark is the Spark Session automatically started by pyspark.
data = spark.createDataFrame(sentences).toDF("text")

# Download the pretrained pipeline from Johnsnowlab's servers
explain_document_pipeline = PretrainedPipeline("explain_document_ml")
# Transform 'data' and store output in a new 'annotations_df' dataframe
annotations_df = explain_document_pipeline.transform(data)

# Show the results
annotations_df.show()
for i in annotations_df:
    print(i)