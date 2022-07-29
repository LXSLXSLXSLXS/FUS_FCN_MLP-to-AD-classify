import pandas as pd
df = pd.read_csv('../lookupcsv/Ground_Truth_Test.csv')
label = df['label'].replace(['AD','NL'],[1,0])
gender = df['gender'].replace(['male','female'],[0,1])

df['label'] = label
df['gender'] = gender
df.to_csv("../lookupcsv/Ground_Truth_Test.csv",index=False,header=True, sep=',')