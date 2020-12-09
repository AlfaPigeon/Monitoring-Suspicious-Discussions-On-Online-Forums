import pandas as pd

df = pd.read_csv("data/labeled_data_t_davidson.csv")[["class","tweet"]]
df = df.replace('\n',' ', regex=True)
df.to_csv('data/labeled_suspicious_text.csv', index=False,sep="|") 
