import pandas as pd

df = pd.read_csv("data/labeled_data_t_davidson.csv")[["class","tweet"]]
df = df.replace('\n',' ', regex=True)

class_ = []
tweet_ = []
c_0 = 0
c_1 = 0
c_2 = 0
for row in df.loc[df["class"] == 0 ].index:
    tweet_.append(df["tweet"][row])
    class_.append(df["class"][row])
    c_0 = c_0 + 1
    if(c_0 == 3000):
        break

for row in df.loc[df["class"] == 1 ].index:
    tweet_.append(df["tweet"][row])
    class_.append(df["class"][row])
    c_1 = c_1 + 1
    if(c_1 == 3000):
        break

for row in df.loc[df["class"] == 2 ].index:
    tweet_.append(df["tweet"][row])
    class_.append(df["class"][row])
    c_2 = c_2 + 1
    if(c_2 == 3000):
        break

df_2 = pd.DataFrame(data= { 'class': class_ , 'tweet': tweet_ })
df_2.to_csv('data/labeled_suspicious_text_balanced.csv', index=False,sep="|") 
