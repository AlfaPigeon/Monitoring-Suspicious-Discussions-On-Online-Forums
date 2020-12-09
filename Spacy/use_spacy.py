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

# ================================================================
epochs = 10
batch_size = 64
data_file = "labeled_suspicious_text.csv"
feature_col = "tweet"
target_col = "class"
vector_size = 300
testing_count = 1000
# ================================================================


data_frame = pd.read_csv("../data/"+data_file, delimiter="|")

print("Prepairing training sets..")
prep_data = []
prep_target = []

for row in [data_frame[feature_col][index] for index in range(len(data_frame[feature_col])) if data_frame[feature_col].notnull()[index] == True]:
    doc_tensor = nlp(row).vector
    prep_data.append(doc_tensor)

for row in [data_frame[target_col][index] for index in range(len(data_frame[feature_col])) if data_frame[feature_col].notnull()[index] == True]:
    if(row != 2):
        prep_target.append(1)
    else:
        prep_target.append(0)

data_size = len([data_frame[target_col][index] for index in range(len(
    data_frame[feature_col])) if data_frame[feature_col].notnull()[index] == True])

# data
print("Forging data..")
data = np.array(prep_data[0:len(prep_data)-testing_count]
                ).reshape((data_size-testing_count, vector_size))
test_data = np.array(prep_data[len(
    prep_data)-testing_count:len(prep_data)]).reshape((testing_count, vector_size))

print("Forging target data..")
target = np.array(prep_target[0:len(prep_target) -
                              testing_count]).reshape((data_size-testing_count))
test_target = np.array(prep_target[len(
    prep_target)-testing_count:len(prep_target)]).reshape((testing_count))
print("#Done")


model = Sequential([
    Dense(100, activation='relu', input_shape=(vector_size,),
          kernel_regularizer=keras.regularizers.l2(l=0.001)),
    Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.001)),
    Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(l=0.001))
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


history = model.fit(
    data,
    target,
    epochs=epochs,
    batch_size=batch_size
)

results = model.evaluate(test_data, test_target, batch_size=batch_size)

model.save('../Models/tf_spacy_oguz')
